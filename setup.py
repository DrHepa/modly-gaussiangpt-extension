#!/usr/bin/env python3
"""Dependency-only, fail-closed setup for the Modly GaussianGPT extension.

Modly invokes this script with one JSON argument. A CLI form exists for local
validation. The script never downloads checkpoints, datasets, or GaussianGPT
source; it only prepares the extension-local virtual environment and probes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXTENSION_ID = "gaussiangpt"
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_NAME = "requirements.txt"
STATUS_RELATIVE_PATH = Path(".modly") / "setup" / "setup-status.json"
LOG_RELATIVE_PATH = Path(".modly") / "setup" / "logs" / "setup.log"
VENV_RELATIVE_PATH = Path("venv")
VENV_PYTHON_RELATIVE_PATH = VENV_RELATIVE_PATH / "bin" / "python"
UPSTREAM_COMMIT = "0615470a5ff359c676408e9ae42036d534d04a43"

PIP_BOOTSTRAP = ("pip==25.1.1", "setuptools==80.9.0", "wheel==0.45.1")
BUILD_DEPS_RELATIVE_PATH = Path(".modly") / "setup" / "build-deps"
OPENBLAS_REPOSITORY = "https://github.com/OpenMathLib/OpenBLAS.git"
OPENBLAS_COMMIT = "6c77e5e314474773a7749357b153caba4ec3817d"
OPENBLAS_VERSION = "0.3.26"
OPENBLAS_BUILD_ARGUMENTS = (
    "TARGET=ARMV8",
    "DYNAMIC_ARCH=1",
    "NUM_THREADS=64",
    "NOFORTRAN=1",
    "NO_SHARED=1",
    "ONLY_CBLAS=1",
)
OPENBLAS_CACHE_SCHEMA = "modly.openblas-static-cache.v1"
MINKOWSKI_REPOSITORY = "https://github.com/alpsaur/MinkowskiEngine.git"
MINKOWSKI_COMMIT = "1a17f71f3158b9e94e90703961695de627f3df08"
MINKOWSKI_PATCHSET = "gaussiangpt-cuda13-compat-v1"
MINKOWSKI_CACHE_SCHEMA = "modly.minkowski-wheel-cache.v1"
MINKOWSKI_BUILD_RISK = (
    "MinkowskiEngine is pinned to commit "
    f"{MINKOWSKI_COMMIT}, removing the mutable-branch identity risk. "
    "Successful compilation and CUDA execution on each toolkit/GPU lane remain "
    "unproven until the strict post-install native probes pass."
)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 1800
COMMAND_TIMEOUT_SECONDS = {
    "python-version-probe": 30,
    "cuda-toolkit-probe": 30,
    "venv-create": 300,
    "pip-bootstrap": 600,
    "torch-install": 1800,
    "python-dependencies-install": 3600,
    "openblas-clone": 900,
    "openblas-checkout": 300,
    "openblas-build": 1800,
    "openblas-install": 600,
    "openblas-link-probe": 300,
    "openblas-run-probe": 60,
    "minkowski-clone": 900,
    "minkowski-checkout": 300,
    "minkowski-build-identity": 120,
    "minkowski-wheel-build": 3600,
    "minkowskiengine-install": 3600,
    "runtime-native-cuda-probe": 600,
    "pip-check": 120,
    "cusparselt-sbsa-probe": 180,
}

FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "download_models", "download_model", "download_weights",
        "bootstrap_models", "bootstrap_weights", "model_dir", "weights_dir",
        "checkpoint", "checkpoint_url", "source_repo", "clone_source",
    }
)
FORBIDDEN_CLI_FLAGS = frozenset(
    {
        "--download-models", "--download-model", "--download-weights",
        "--bootstrap-models", "--bootstrap-weights", "--model-dir",
        "--weights-dir", "--checkpoint", "--checkpoint-url", "--source-repo",
        "--clone-source",
    }
)
REQUIRED_RUNTIME_IMPORTS = (
    "torch", "torchvision", "lightning", "omegaconf", "hydra",
    "MinkowskiEngine", "vector_quantize_pytorch", "numpy", "plyfile",
    "einops", "imageio", "gsplat", "torchmetrics", "lpips", "pytorch3d", "PIL",
)

CUSPARSELT_PIP_CHECK_OUTPUT = (
    "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform\n"
)
CUSPARSELT_TORCH_REQUIREMENT = (
    'nvidia-cusparselt-cu13==0.8.0; platform_system == "Linux"'
)
CUSPARSELT_TORCH_DISTRIBUTION_VERSION = "2.9.1+cu130"
CUSPARSELT_WHEEL_TAG = "py3-none-manylinux2014_sbsa"
CUSPARSELT_LIBRARY_PATH = "nvidia/cusparselt/lib/libcusparseLt.so.0"
CUSPARSELT_REQUIRED_SYMBOLS = (
    "cusparseLtGetProperty",
    "cusparseLtInit",
    "cusparseLtDestroy",
)
CUSPARSELT_PROBE_SCHEMA = "modly.cusparselt-sbsa-probe.v1"
CUDA13_NVCC_FLAGS = (
    "--device-entity-has-hidden-visibility=false",
    "-static-global-template-stub=false",
)


@dataclass(frozen=True)
class RuntimeLane:
    lane_id: str
    system: str
    machine: str
    cuda_code: int
    python_version: tuple[int, int]
    torch_version: str
    torchvision_version: str
    torch_index_url: str
    expected_torch_cuda: str
    support_state: str
    torch_cuda_arch_list: str
    evidence: str
    risks: tuple[str, ...] = ()


LANES: dict[tuple[str, str, int], RuntimeLane] = {
    ("linux", "aarch64", 130): RuntimeLane(
        lane_id="linux-aarch64-cp311-cu130-experimental",
        system="linux",
        machine="aarch64",
        cuda_code=130,
        python_version=(3, 11),
        torch_version="2.9.1",
        torchvision_version="0.24.1",
        torch_index_url="https://download.pytorch.org/whl/cu130",
        expected_torch_cuda="13.0",
        support_state="experimental-fail-closed",
        torch_cuda_arch_list="12.1",
        evidence=(
            "Packaged Modly provides CPython 3.11.9. The official PyTorch cu130 "
            "index provides the selected torch 2.9.1 and torchvision 0.24.1 "
            "lane. GaussianGPT native dependencies on aarch64/CUDA 13.0 remain "
            "unvalidated and setup stays fail-closed until every probe passes."
        ),
        risks=(
            "The complete CPython 3.11 aarch64 CUDA 13.0 native dependency lane is unvalidated.",
            "MinkowskiEngine, PyTorch3D and gsplat must each execute a CUDA probe before readiness.",
            MINKOWSKI_BUILD_RISK,
        ),
    ),
    ("linux", "aarch64", 128): RuntimeLane(
        lane_id="linux-aarch64-cp311-cu128-experimental",
        system="linux",
        machine="aarch64",
        cuda_code=128,
        python_version=(3, 11),
        torch_version="2.7.1",
        torchvision_version="0.22.1",
        torch_index_url="https://download.pytorch.org/whl/cu128",
        expected_torch_cuda="12.8",
        support_state="experimental-fail-closed",
        torch_cuda_arch_list="12.0",
        evidence=(
            "Packaged Modly provides CPython 3.11.9. The official PyTorch cu128 "
            "index provides the selected torch 2.7.1 and torchvision 0.22.1 "
            "lane. A separate Python 3.10/Torch 2.7.1+cu128 GB10 probe executed "
            "a CUDA tensor, but does not prove the complete CPython 3.11 native "
            "dependency lane."
        ),
        risks=(
            "GB10 reports sm121 while torch 2.7.1+cu128 advertises sm120.",
            "The complete CPython 3.11 aarch64 CUDA 12.8 native dependency lane is unvalidated.",
            MINKOWSKI_BUILD_RISK,
        ),
    ),
}


@dataclass(frozen=True)
class SetupConfig:
    python_exe: str
    ext_dir: Path
    gpu_sm: int | None = None
    cuda_version: int | None = None
    validate_only: bool = False
    no_install: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def requirements_path(self) -> Path:
        return self.ext_dir / REQUIREMENTS_NAME

    @property
    def venv_dir(self) -> Path:
        return self.ext_dir / VENV_RELATIVE_PATH

    @property
    def venv_python(self) -> Path:
        return self.ext_dir / VENV_PYTHON_RELATIVE_PATH

    @property
    def status_path(self) -> Path:
        return self.ext_dir / STATUS_RELATIVE_PATH

    @property
    def log_path(self) -> Path:
        return self.ext_dir / LOG_RELATIVE_PATH

    @property
    def build_deps_dir(self) -> Path:
        return self.ext_dir / BUILD_DEPS_RELATIVE_PATH


class SetupContractError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        step: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.step = step
        self.details = dict(details or {})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(message: str) -> None:
    print(f"[setup:gaussiangpt] {message}", file=sys.stderr, flush=True)


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise SetupContractError("invalid_boolean", f"{name} must be a boolean.", step="parse")


def parse_optional_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise SetupContractError("invalid_integer", f"{name} must be an integer.", step="parse")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SetupContractError("invalid_integer", f"{name} must be an integer.", step="parse") from exc


def reject_forbidden_payload(payload: Mapping[str, Any]) -> None:
    present = sorted(FORBIDDEN_PAYLOAD_KEYS.intersection(payload))
    if present:
        raise SetupContractError(
            "model_download_forbidden",
            "GaussianGPT setup rejects model/source download controls: "
            + ", ".join(present)
            + ". Model assets are managed exclusively by Modly's HTTPS plan, "
            "and GaussianGPT source is already vendored.",
            step="parse",
            details={"forbidden_keys": present},
        )


def parse_setup_config(argv: Sequence[str]) -> SetupConfig:
    args = list(argv)
    if len(args) == 1 and args[0].lstrip().startswith("{"):
        try:
            payload = json.loads(args[0])
        except json.JSONDecodeError as exc:
            raise SetupContractError("invalid_json", f"Modly setup payload is invalid JSON: {exc}", step="parse") from exc
        if not isinstance(payload, dict):
            raise SetupContractError("invalid_json", "Modly setup payload must be a JSON object.", step="parse")
        reject_forbidden_payload(payload)
        return SetupConfig(
            python_exe=str(payload.get("python_exe") or sys.executable),
            ext_dir=Path(payload.get("ext_dir") or SCRIPT_DIR).expanduser().resolve(),
            gpu_sm=parse_optional_int(payload.get("gpu_sm"), "gpu_sm"),
            cuda_version=parse_optional_int(payload.get("cuda_version"), "cuda_version"),
            validate_only=parse_bool(payload.get("validate_only"), "validate_only"),
            no_install=parse_bool(payload.get("no_install"), "no_install"),
            payload=dict(payload),
        )
    if len(args) > 1 and any(item.lstrip().startswith("{") for item in args):
        raise SetupContractError(
            "invalid_arguments",
            "The Modly contract accepts exactly one JSON argument.",
            step="parse",
        )

    forbidden_cli = [
        token.split("=", 1)[0]
        for token in args
        if token.split("=", 1)[0] in FORBIDDEN_CLI_FLAGS
    ]
    if forbidden_cli:
        raise SetupContractError(
            "model_download_forbidden",
            "GaussianGPT setup rejects model/source download flags: "
            + ", ".join(sorted(set(forbidden_cli))),
            step="parse",
        )

    parser = argparse.ArgumentParser(description="Prepare or validate the GaussianGPT dependency runtime.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--ext-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--gpu-sm", type=int, default=None)
    parser.add_argument("--cuda-version", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        raise SetupContractError("invalid_arguments", "Invalid GaussianGPT setup arguments.", step="parse") from exc
    return SetupConfig(
        python_exe=str(parsed.python_exe),
        ext_dir=Path(parsed.ext_dir).expanduser().resolve(),
        gpu_sm=parsed.gpu_sm,
        cuda_version=parsed.cuda_version,
        validate_only=bool(parsed.validate_only),
        no_install=bool(parsed.no_install),
    )


def append_log(config: SetupConfig, message: str) -> None:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message.rstrip()}\n")


def resolve_python(executable: str) -> Path:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
    else:
        found = shutil.which(executable)
        if not found:
            raise SetupContractError("python_not_found", f"Python executable was not found: {executable}", step="preflight")
        resolved = Path(found).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SetupContractError("python_not_executable", f"Python executable is not runnable: {resolved}", step="preflight")
    return resolved


def run_command(
    config: SetupConfig,
    command: Sequence[str],
    *,
    step: str,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    timeout_seconds: int | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command_list = [str(item) for item in command]
    working_directory = (config.ext_dir if cwd is None else cwd).resolve()
    try:
        working_directory.relative_to(config.ext_dir.resolve())
    except ValueError as exc:
        raise SetupContractError(
            "command_cwd_escape",
            f"{step} working directory escapes ext_dir: {working_directory}",
            step=step,
        ) from exc
    if not working_directory.is_dir():
        raise SetupContractError(
            "command_cwd_missing",
            f"{step} working directory is missing: {working_directory}",
            step=step,
        )
    timeout = int(
        timeout_seconds
        if timeout_seconds is not None
        else COMMAND_TIMEOUT_SECONDS.get(step, DEFAULT_COMMAND_TIMEOUT_SECONDS)
    )
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks: list[str] = []
    command_text = shlex.join(command_list)

    try:
        process = subprocess.Popen(
            command_list,
            cwd=str(working_directory),
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        raise SetupContractError(
            "command_start_failed",
            f"{step} could not start: {exc}",
            step=step,
            details={"command": command_list},
        ) from exc

    def stream_output() -> None:
        assert process.stdout is not None
        with config.log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"[{utc_now()}] {step}: cwd={working_directory} $ {command_text} "
                f"(timeout={timeout}s)\n"
            )
            log_handle.flush()
            for chunk in iter(process.stdout.readline, ""):
                output_chunks.append(chunk)
                print(chunk, file=sys.stderr, end="", flush=True)
                log_handle.write(chunk)
                log_handle.flush()

    reader = threading.Thread(
        target=stream_output,
        name=f"gaussiangpt-setup-{step}",
        daemon=True,
    )
    reader.start()
    timed_out = False
    started_at = time.monotonic()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        returncode = process.returncode
    finally:
        reader.join(timeout=10)
        if process.stdout is not None:
            process.stdout.close()

    elapsed = time.monotonic() - started_at
    output = "".join(output_chunks)
    append_log(
        config,
        f"{step}: exit={returncode} elapsed={elapsed:.3f}s"
        + (" timed_out=true" if timed_out else ""),
    )
    completed = subprocess.CompletedProcess(
        command_list,
        returncode,
        stdout=output,
        stderr=None,
    )
    if timed_out:
        raise SetupContractError(
            "command_timeout",
            f"{step} exceeded its {timeout}-second process timeout and was terminated. "
            f"See {config.log_path}.",
            step=step,
            details={
                "command": command_list,
                "timeout_seconds": timeout,
                "returncode": returncode,
            },
        )
    if check and returncode != 0:
        raise SetupContractError(
            "command_failed",
            f"{step} failed with exit {returncode}. See {config.log_path}.",
            step=step,
            details={"command": command_list, "returncode": returncode},
        )
    return completed


def validate_extension_anchor(config: SetupConfig) -> dict[str, Any]:
    if not config.ext_dir.is_dir():
        raise SetupContractError("invalid_ext_dir", f"ext_dir does not exist: {config.ext_dir}", step="preflight")
    manifest_path = config.ext_dir / "manifest.json"
    upstream_path = config.ext_dir / "UPSTREAM.json"
    vendor_root = config.ext_dir / "vendor" / "GaussianGPT"
    required_paths = (
        config.ext_dir / "setup.py",
        config.requirements_path,
        manifest_path,
        upstream_path,
        vendor_root / "LICENSE",
        vendor_root / "README.md",
        vendor_root / "model" / "gaussian_gpt.py",
        vendor_root / "model" / "gaussian_vqvae.py",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SetupContractError(
            "extension_incomplete",
            "Extension source is incomplete: " + ", ".join(missing),
            step="preflight",
            details={"missing": missing},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupContractError("invalid_extension_metadata", f"Cannot validate extension metadata: {exc}", step="preflight") from exc
    if manifest.get("id") != EXTENSION_ID:
        raise SetupContractError("wrong_extension", f"ext_dir manifest id must be {EXTENSION_ID!r}.", step="preflight")
    if upstream.get("commit") != UPSTREAM_COMMIT:
        raise SetupContractError("wrong_upstream_commit", f"Vendored upstream metadata must pin commit {UPSTREAM_COMMIT}.", step="preflight")
    try:
        vendor_root.resolve().relative_to(config.ext_dir)
    except ValueError as exc:
        raise SetupContractError("vendor_escape", "Vendored source must resolve inside ext_dir.", step="preflight") from exc
    return {
        "ext_dir": str(config.ext_dir),
        "manifest": str(manifest_path),
        "requirements": str(config.requirements_path),
        "vendor_root": str(vendor_root.resolve()),
        "upstream_commit": UPSTREAM_COMMIT,
    }


def probe_python_version(config: SetupConfig, python_exe: Path) -> tuple[int, int, int]:
    process = run_command(
        config,
        [str(python_exe), "-B", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
        step="python-version-probe",
    )
    try:
        values = json.loads(process.stdout.strip().splitlines()[-1])
        return int(values[0]), int(values[1]), int(values[2])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SetupContractError("python_probe_invalid", f"Could not parse Python version from {python_exe}.", step="python-version-probe") from exc


def normalized_platform() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x64"}:
        machine = "x86_64"
    if machine in {"arm64", "armv8l"}:
        machine = "aarch64"
    return system, machine


def toolkit_release_code(release: str) -> int:
    match = re.fullmatch(r"(\d+)\.(\d+)", release)
    if not match:
        raise SetupContractError(
            "invalid_cuda_toolkit_release",
            f"Cannot normalize CUDA toolkit release {release!r}.",
            step="lane-selection",
        )
    return int(match.group(1)) * 10 + int(match.group(2))


def select_lane(
    config: SetupConfig,
    python_version: tuple[int, int, int],
    toolkits: Sequence[Mapping[str, Any]],
) -> tuple[RuntimeLane, dict[str, Any]]:
    system, machine = normalized_platform()
    for toolkit_value in toolkits:
        toolkit = dict(toolkit_value)
        cuda_code = toolkit_release_code(str(toolkit["release"]))
        lane = LANES.get((system, machine, cuda_code))
        if lane is None:
            continue
        if python_version[:2] != lane.python_version:
            continue
        return lane, toolkit

    supported = ", ".join(item.lane_id for item in LANES.values())
    actual_releases = [str(item.get("release")) for item in toolkits]
    raise SetupContractError(
        "unsupported_runtime_lane",
        f"No audited GaussianGPT lane matches {system}/{machine}, Python "
        f"{python_version[0]}.{python_version[1]}, and probed CUDA toolkit(s) "
        f"{actual_releases}. Supported lanes: {supported}. "
        f"The Electron cuda_version value {config.cuda_version!r} is only a "
        "driver-derived hint and was not used as toolkit proof.",
        step="lane-selection",
        details={
            "system": system,
            "machine": machine,
            "python_version": list(python_version),
            "driver_cuda_hint": config.cuda_version,
            "probed_toolkits": [dict(item) for item in toolkits],
        },
    )


def probe_cuda_toolkits(config: SetupConfig) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    configured = os.environ.get("CUDA_HOME")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path("/usr/local/cuda"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))

    seen: set[str] = set()
    toolkits: list[dict[str, Any]] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        nvcc = resolved / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        process = run_command(
            config,
            [str(nvcc), "--version"],
            step="cuda-toolkit-probe",
            check=False,
            timeout_seconds=30,
        )
        match = re.search(r"release\s+(\d+\.\d+)", process.stdout)
        release = match.group(1) if match else None
        if process.returncode == 0 and release is not None:
            toolkits.append(
                {
                    "cuda_home": str(resolved),
                    "nvcc": str(nvcc),
                    "release": release,
                    "driver_cuda_hint": config.cuda_version,
                }
            )
    if toolkits:
        return toolkits
    raise SetupContractError(
        "cuda_toolkit_unavailable",
        "No CUDA toolkit with a runnable nvcc was found via CUDA_HOME, "
        "/usr/local/cuda, or /usr/local/cuda-*. The Electron cuda_version "
        f"value {config.cuda_version!r} is a driver hint, not toolkit evidence.",
        step="cuda-toolkit-probe",
        details={"driver_cuda_hint": config.cuda_version},
    )


def create_extension_venv(config: SetupConfig, python_exe: Path) -> Path:
    if config.venv_python.is_file():
        append_log(config, f"venv: reusing {config.venv_python}")
        return config.venv_python
    if config.venv_dir.exists():
        raise SetupContractError(
            "partial_venv",
            f"{config.venv_dir} exists but {config.venv_python} is missing. Remove the incomplete venv before retrying.",
            step="venv-create",
        )
    run_command(config, [str(python_exe), "-m", "venv", str(config.venv_dir)], step="venv-create")
    if not config.venv_python.is_file():
        raise SetupContractError("venv_python_missing", f"venv creation returned success but {config.venv_python} is missing.", step="venv-create")
    return config.venv_python


def merge_required_nvcc_flags(
    existing: str,
    required: Sequence[str],
) -> str:
    required_set = set(required)
    seen_required: set[str] = set()
    merged: list[str] = []
    for token in existing.split():
        if token in required_set:
            if token in seen_required:
                continue
            seen_required.add(token)
        merged.append(token)
    merged.extend(flag for flag in required if flag not in seen_required)
    return " ".join(merged)


def native_build_environment(lane: RuntimeLane, toolkit: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_HOME"] = str(toolkit["cuda_home"])
    env["TORCH_CUDA_ARCH_LIST"] = lane.torch_cuda_arch_list
    env.setdefault("MAX_JOBS", "2")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PYTHONNOUSERSITE", "1")
    if lane.cuda_code >= 130:
        env["NVCC_FLAGS"] = merge_required_nvcc_flags(
            env.get("NVCC_FLAGS", ""),
            CUDA13_NVCC_FLAGS,
        )
    return env


MINKOWSKI_COORDINATE_OLD = """template <typename coordinate_type, typename map_type>
struct find_coordinate
    : public thrust::unary_function<uint32_t,
                                    thrust::pair<uint32_t, uint32_t>> {"""
MINKOWSKI_COORDINATE_NEW = """template <typename coordinate_type, typename map_type>
struct find_coordinate {"""
MINKOWSKI_PREFETCH_INSERT_ANCHOR = """#include <type_traits>

namespace {
template <std::size_t N>"""
MINKOWSKI_PREFETCH_HELPER = """inline cudaError_t cuda_mem_prefetch_async_compat(
    const void *ptr, size_t count, int dev_id, cudaStream_t stream) {
#if CUDART_VERSION >= 13000
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id = dev_id;
  return cudaMemPrefetchAsync(ptr, count, location, 0, stream);
#else
  return cudaMemPrefetchAsync(ptr, count, dev_id, stream);
#endif
}
"""
MINKOWSKI_LEGACY_PREFETCH_CALL = "cudaMemPrefetchAsync("
MINKOWSKI_COMPAT_PREFETCH_CALL = "cuda_mem_prefetch_async_compat("


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_path_inside(root: Path, path: Path, *, step: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SetupContractError(
            "build_dependency_path_escape",
            f"{step} path escapes {resolved_root}: {resolved_path}",
            step=step,
        ) from exc
    return resolved_path


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    mode = path.stat().st_mode if path.exists() else None
    try:
        temporary.write_text(text, encoding="utf-8")
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def read_json_object(path: Path, *, code: str, step: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupContractError(
            code,
            f"Cannot read validated JSON at {path}: {exc}",
            step=step,
        ) from exc
    if not isinstance(value, dict):
        raise SetupContractError(code, f"Expected a JSON object at {path}.", step=step)
    return value


def remove_build_tree(config: SetupConfig, path: Path, *, step: str) -> None:
    resolved = require_path_inside(config.build_deps_dir, path, step=step)
    if resolved == config.build_deps_dir.resolve():
        raise SetupContractError(
            "unsafe_build_dependency_cleanup",
            "Refusing to remove the build-deps root.",
            step=step,
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def invalidate_cache_directory(
    config: SetupConfig,
    cache_dir: Path,
    *,
    step: str,
    reason: str,
) -> None:
    require_path_inside(config.build_deps_dir, cache_dir, step=step)
    quarantine = cache_dir.with_name(
        f".invalid-{cache_dir.name}-{uuid.uuid4().hex}"
    )
    append_log(config, f"{step}: rebuilding invalid cache {cache_dir}: {reason}")
    os.replace(cache_dir, quarantine)
    remove_build_tree(config, quarantine, step=step)


def promote_staging_directory(
    config: SetupConfig,
    staging: Path,
    destination: Path,
    *,
    step: str,
) -> None:
    require_path_inside(config.build_deps_dir, staging, step=step)
    require_path_inside(config.build_deps_dir, destination, step=step)
    if destination.exists():
        raise SetupContractError(
            "build_dependency_promotion_conflict",
            f"Refusing to overwrite an existing cache during atomic promotion: {destination}",
            step=step,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)


def checkout_pinned_repository(
    config: SetupConfig,
    *,
    repository: str,
    commit: str,
    destination: Path,
    clone_step: str,
    checkout_step: str,
) -> dict[str, str]:
    require_path_inside(config.build_deps_dir, destination, step=clone_step)
    if destination.exists():
        raise SetupContractError(
            "dependency_checkout_exists",
            f"Pinned dependency checkout destination already exists: {destination}",
            step=clone_step,
        )
    run_command(
        config,
        ["git", "clone", "--no-checkout", repository, str(destination)],
        step=clone_step,
    )
    run_command(
        config,
        ["git", "-C", str(destination), "checkout", "--detach", commit],
        step=checkout_step,
    )
    head = run_command(
        config,
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        step=checkout_step,
    ).stdout.strip()
    remote = run_command(
        config,
        ["git", "-C", str(destination), "remote", "get-url", "origin"],
        step=checkout_step,
    ).stdout.strip()
    dirty = run_command(
        config,
        [
            "git", "-C", str(destination), "status", "--porcelain=v1",
            "--untracked-files=no",
        ],
        step=checkout_step,
    ).stdout
    if head != commit or remote != repository or dirty != "":
        raise SetupContractError(
            "dependency_checkout_mismatch",
            f"Pinned dependency checkout validation failed for {repository}.",
            step=checkout_step,
            details={
                "expected_commit": commit,
                "actual_commit": head,
                "expected_repository": repository,
                "actual_repository": remote,
                "tracked_changes": dirty,
            },
        )
    return {"repository": remote, "commit": head, "tracked_changes": dirty}


def openblas_cache_identity() -> dict[str, Any]:
    return {
        "repository": OPENBLAS_REPOSITORY,
        "commit": OPENBLAS_COMMIT,
        "version": OPENBLAS_VERSION,
        "build_arguments": list(OPENBLAS_BUILD_ARGUMENTS),
        "make_jobs": 2,
    }


def inspect_openblas_prefix(prefix: Path, *, step: str) -> dict[str, Any]:
    resolved_prefix = prefix.resolve()
    include_dir = resolved_prefix / "include"
    library_dir = resolved_prefix / "lib"
    header = include_dir / "cblas.h"
    if (
        not header.is_file()
        or header.is_symlink()
        or not library_dir.is_dir()
    ):
        raise SetupContractError(
            "openblas_artifact_missing",
            f"Static OpenBLAS install is incomplete at {resolved_prefix}.",
            step=step,
        )

    shared = sorted(
        {
            item.name
            for pattern in ("libopenblas*.so*", "libopenblas*.dylib")
            for item in library_dir.glob(pattern)
        }
    )
    if shared:
        raise SetupContractError(
            "openblas_shared_artifact_present",
            f"Static-only OpenBLAS cache contains shared libraries: {shared}",
            step=step,
            details={"shared_libraries": shared},
        )

    candidates = sorted(library_dir.glob("libopenblas*.a"))
    actual_archives: dict[str, Path] = {}
    for candidate in candidates:
        resolved = require_path_inside(resolved_prefix, candidate, step=step)
        if not resolved.is_file() or resolved.is_symlink():
            raise SetupContractError(
                "openblas_archive_invalid",
                f"OpenBLAS archive target is not a regular file: {resolved}",
                step=step,
            )
        actual_archives[str(resolved)] = resolved
    if len(actual_archives) != 1:
        raise SetupContractError(
            "openblas_archive_ambiguous",
            "Static OpenBLAS cache must resolve to exactly one archive target.",
            step=step,
            details={
                "candidates": [str(item) for item in candidates],
                "targets": sorted(actual_archives),
            },
        )
    archive = next(iter(actual_archives.values()))
    return {
        "prefix": ".",
        "header": str(header.relative_to(resolved_prefix)),
        "archive": str(archive.relative_to(resolved_prefix)),
        "archive_sha256": file_sha256(archive),
        "archive_size": archive.stat().st_size,
        "shared_libraries": [],
    }


OPENBLAS_LINK_PROBE_SOURCE = r"""
#include <cblas.h>

int main(void) {
  double x[3] = {1.0, 2.0, 3.0};
  double y[3] = {4.0, 5.0, 6.0};
  cblas_daxpy(3, 2.0, x, 1, y, 1);
  return (y[0] == 6.0 && y[1] == 9.0 && y[2] == 12.0) ? 0 : 41;
}
"""


def run_openblas_link_probe(
    config: SetupConfig,
    prefix: Path,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    compiler_value = shutil.which("cc") or shutil.which("gcc")
    if compiler_value is None:
        raise SetupContractError(
            "c_compiler_unavailable",
            "A C compiler is required to build the extension-local OpenBLAS dependency.",
            step="openblas-link-probe",
        )
    compiler = Path(compiler_value).resolve()
    archive = require_path_inside(
        prefix,
        prefix / str(artifact["archive"]),
        step="openblas-link-probe",
    )
    probe_dir = config.build_deps_dir / f".openblas-probe-{uuid.uuid4().hex}"
    probe_dir.mkdir(parents=True, exist_ok=False)
    source = probe_dir / "probe.c"
    executable = probe_dir / "probe"
    try:
        source.write_text(OPENBLAS_LINK_PROBE_SOURCE, encoding="utf-8")
        run_command(
            config,
            [
                str(compiler),
                "-std=c11",
                "-I",
                str(prefix / "include"),
                str(source),
                str(archive),
                "-lpthread",
                "-lm",
                "-ldl",
                "-o",
                str(executable),
            ],
            step="openblas-link-probe",
            cwd=probe_dir,
        )
        if not executable.is_file() or executable.is_symlink():
            raise SetupContractError(
                "openblas_probe_binary_missing",
                "OpenBLAS link command succeeded without a regular probe binary.",
                step="openblas-link-probe",
            )
        run_command(
            config,
            [str(executable)],
            step="openblas-run-probe",
            cwd=probe_dir,
        )
        return {
            "compiler": str(compiler),
            "archive_sha256": str(artifact["archive_sha256"]),
            "linked": True,
            "executed": True,
        }
    finally:
        remove_build_tree(config, probe_dir, step="openblas-link-probe")


def validate_openblas_cache(
    config: SetupConfig,
    cache_dir: Path,
    *,
    expected_cache_key: str,
    run_probe: bool,
) -> dict[str, Any]:
    require_path_inside(config.build_deps_dir, cache_dir, step="openblas-cache-validation")
    provenance = read_json_object(
        cache_dir / "provenance.json",
        code="openblas_provenance_invalid",
        step="openblas-cache-validation",
    )
    expected_identity = openblas_cache_identity()
    if (
        provenance.get("schema") != OPENBLAS_CACHE_SCHEMA
        or provenance.get("cache_key") != expected_cache_key
        or provenance.get("identity") != expected_identity
    ):
        raise SetupContractError(
            "openblas_provenance_mismatch",
            "OpenBLAS cache provenance does not match the immutable build plan.",
            step="openblas-cache-validation",
            details={"provenance": provenance},
        )
    artifact = inspect_openblas_prefix(
        cache_dir / "prefix",
        step="openblas-cache-validation",
    )
    recorded_artifact = provenance.get("artifact")
    if not isinstance(recorded_artifact, dict) or any(
        recorded_artifact.get(key) != artifact.get(key)
        for key in (
            "prefix", "header", "archive", "archive_sha256",
            "archive_size", "shared_libraries",
        )
    ):
        raise SetupContractError(
            "openblas_artifact_provenance_mismatch",
            "OpenBLAS artifact hash, size, or path differs from provenance.",
            step="openblas-cache-validation",
        )
    probe = (
        run_openblas_link_probe(config, cache_dir / "prefix", artifact)
        if run_probe
        else dict(provenance.get("functional_probe") or {})
    )
    return {
        "schema": OPENBLAS_CACHE_SCHEMA,
        "cache_key": expected_cache_key,
        "cache_dir": str(cache_dir),
        "repository": OPENBLAS_REPOSITORY,
        "commit": OPENBLAS_COMMIT,
        "version": OPENBLAS_VERSION,
        "build_arguments": list(OPENBLAS_BUILD_ARGUMENTS),
        "artifact": artifact,
        "functional_probe": probe,
    }


def ensure_openblas_dependency(
    config: SetupConfig,
    lane: RuntimeLane,
) -> dict[str, Any]:
    if lane.machine != "aarch64":
        raise SetupContractError(
            "openblas_target_mismatch",
            f"The pinned OpenBLAS build plan only supports aarch64, not {lane.machine}.",
            step="openblas-build",
        )
    identity = openblas_cache_identity()
    cache_key = canonical_sha256(identity)
    cache_dir = config.build_deps_dir / "openblas" / cache_key
    config.build_deps_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        try:
            return validate_openblas_cache(
                config,
                cache_dir,
                expected_cache_key=cache_key,
                run_probe=True,
            )
        except SetupContractError as exc:
            invalidate_cache_directory(
                config,
                cache_dir,
                step="openblas-cache-validation",
                reason=str(exc),
            )

    staging = config.build_deps_dir / f".staging-openblas-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    source = staging / "source"
    prefix = staging / "prefix"
    try:
        checkout_pinned_repository(
            config,
            repository=OPENBLAS_REPOSITORY,
            commit=OPENBLAS_COMMIT,
            destination=source,
            clone_step="openblas-clone",
            checkout_step="openblas-checkout",
        )
        run_command(
            config,
            [
                "make", "-C", str(source), "-j2",
                *OPENBLAS_BUILD_ARGUMENTS,
            ],
            step="openblas-build",
        )
        run_command(
            config,
            [
                "make", "-C", str(source),
                *OPENBLAS_BUILD_ARGUMENTS,
                f"PREFIX={prefix}",
                "install",
            ],
            step="openblas-install",
        )
        artifact = inspect_openblas_prefix(prefix, step="openblas-install")
        functional_probe = run_openblas_link_probe(config, prefix, artifact)
        remove_build_tree(config, source, step="openblas-install")
        atomic_write_json(
            staging / "provenance.json",
            {
                "schema": OPENBLAS_CACHE_SCHEMA,
                "cache_key": cache_key,
                "identity": identity,
                "artifact": artifact,
                "functional_probe": functional_probe,
            },
        )
        promote_staging_directory(
            config,
            staging,
            cache_dir,
            step="openblas-cache-promotion",
        )
    finally:
        if staging.exists():
            remove_build_tree(config, staging, step="openblas-cache-cleanup")
    return validate_openblas_cache(
        config,
        cache_dir,
        expected_cache_key=cache_key,
        run_probe=False,
    )


def apply_minkowski_compatibility_patches(
    source: Path,
    *,
    cuda_code: int,
) -> dict[str, Any]:
    coordinate_path = source / "src" / "coordinate_map_functors.cuh"
    prefetch_path = source / "src" / "3rdparty" / "concurrent_unordered_map.cuh"
    if not coordinate_path.is_file() or not prefetch_path.is_file():
        raise SetupContractError(
            "minkowski_patch_file_missing",
            "Pinned MinkowskiEngine checkout lacks the expected CUDA source files.",
            step="minkowski-patch",
        )
    if cuda_code < 130:
        return {
            "patchset": "none",
            "applied": False,
            "cuda_code": cuda_code,
            "files": [],
        }

    coordinate_original = coordinate_path.read_text(encoding="utf-8")
    prefetch_original = prefetch_path.read_text(encoding="utf-8")
    counts = {
        "coordinate_old": coordinate_original.count(MINKOWSKI_COORDINATE_OLD),
        "coordinate_new": coordinate_original.count(MINKOWSKI_COORDINATE_NEW),
        "prefetch_anchor": prefetch_original.count(MINKOWSKI_PREFETCH_INSERT_ANCHOR),
        "legacy_prefetch_calls": prefetch_original.count(
            MINKOWSKI_LEGACY_PREFETCH_CALL
        ),
        "compat_helper": prefetch_original.count(
            "inline cudaError_t cuda_mem_prefetch_async_compat("
        ),
    }
    expected = {
        "coordinate_old": 1,
        "coordinate_new": 0,
        "prefetch_anchor": 1,
        "legacy_prefetch_calls": 3,
        "compat_helper": 0,
    }
    if counts != expected:
        raise SetupContractError(
            "minkowski_patch_anchor_mismatch",
            "Pinned MinkowskiEngine CUDA 13 patch anchors do not match exactly.",
            step="minkowski-patch",
            details={"expected": expected, "actual": counts},
        )

    coordinate_patched = coordinate_original.replace(
        MINKOWSKI_COORDINATE_OLD,
        MINKOWSKI_COORDINATE_NEW,
        1,
    )
    prefetch_patched = prefetch_original.replace(
        MINKOWSKI_LEGACY_PREFETCH_CALL,
        MINKOWSKI_COMPAT_PREFETCH_CALL,
    )
    prefetch_patched = prefetch_patched.replace(
        MINKOWSKI_PREFETCH_INSERT_ANCHOR,
        (
            "#include <type_traits>\n\nnamespace {\n"
            + MINKOWSKI_PREFETCH_HELPER
            + "\ntemplate <std::size_t N>"
        ),
        1,
    )
    post_counts = {
        "coordinate_old": coordinate_patched.count(MINKOWSKI_COORDINATE_OLD),
        "coordinate_new": coordinate_patched.count(MINKOWSKI_COORDINATE_NEW),
        "compat_calls": prefetch_patched.count(MINKOWSKI_COMPAT_PREFETCH_CALL),
        "cuda_runtime_calls": prefetch_patched.count(
            MINKOWSKI_LEGACY_PREFETCH_CALL
        ),
        "compat_helper": prefetch_patched.count(
            "inline cudaError_t cuda_mem_prefetch_async_compat("
        ),
    }
    expected_post = {
        "coordinate_old": 0,
        "coordinate_new": 1,
        "compat_calls": 4,
        "cuda_runtime_calls": 2,
        "compat_helper": 1,
    }
    if post_counts != expected_post:
        raise SetupContractError(
            "minkowski_patch_result_mismatch",
            "MinkowskiEngine CUDA 13 patch result failed exact validation.",
            step="minkowski-patch",
            details={"expected": expected_post, "actual": post_counts},
        )

    # Both files are fully validated before either atomic replacement. The
    # checkout itself remains private staging until the wheel is complete.
    atomic_write_text(coordinate_path, coordinate_patched)
    atomic_write_text(prefetch_path, prefetch_patched)
    return {
        "patchset": MINKOWSKI_PATCHSET,
        "applied": True,
        "cuda_code": cuda_code,
        "pre_counts": counts,
        "post_counts": post_counts,
        "files": [
            {
                "path": "src/coordinate_map_functors.cuh",
                "before_sha256": hashlib.sha256(
                    coordinate_original.encode("utf-8")
                ).hexdigest(),
                "after_sha256": hashlib.sha256(
                    coordinate_patched.encode("utf-8")
                ).hexdigest(),
            },
            {
                "path": "src/3rdparty/concurrent_unordered_map.cuh",
                "before_sha256": hashlib.sha256(
                    prefetch_original.encode("utf-8")
                ).hexdigest(),
                "after_sha256": hashlib.sha256(
                    prefetch_patched.encode("utf-8")
                ).hexdigest(),
            },
        ],
    }


def resolve_cuda_cccl_include(
    lane: RuntimeLane,
    toolkit: Mapping[str, Any],
) -> Path | None:
    if lane.cuda_code < 130:
        return None
    cuda_home = Path(str(toolkit["cuda_home"])).resolve()
    cccl = (
        cuda_home
        / "targets"
        / "sbsa-linux"
        / "include"
        / "cccl"
    ).resolve()
    try:
        cccl.relative_to(cuda_home)
    except ValueError as exc:
        raise SetupContractError(
            "cuda_cccl_path_escape",
            f"CUDA CCCL include path escapes CUDA_HOME: {cccl}",
            step="minkowski-build-environment",
        ) from exc
    thrust_header = cccl / "thrust" / "host_vector.h"
    if not thrust_header.is_file() or thrust_header.is_symlink():
        raise SetupContractError(
            "cuda_cccl_header_missing",
            f"CUDA 13 CCCL header is missing: {thrust_header}",
            step="minkowski-build-environment",
        )
    return cccl


def prepend_search_paths(paths: Sequence[Path], existing: str) -> str:
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def minkowski_build_environment(
    lane: RuntimeLane,
    toolkit: Mapping[str, Any],
    openblas_prefix: Path,
) -> tuple[dict[str, str], Path | None]:
    include_dir = openblas_prefix / "include"
    library_dir = openblas_prefix / "lib"
    if not (include_dir / "cblas.h").is_file() or not library_dir.is_dir():
        raise SetupContractError(
            "openblas_build_environment_invalid",
            f"OpenBLAS build prefix is incomplete: {openblas_prefix}",
            step="minkowski-build-environment",
        )
    env = native_build_environment(lane, toolkit)
    cccl = resolve_cuda_cccl_include(lane, toolkit)
    cpath_entries = [include_dir]
    if cccl is not None:
        cpath_entries.append(cccl)
    env["CPATH"] = prepend_search_paths(
        cpath_entries,
        env.get("CPATH", ""),
    )
    env["LIBRARY_PATH"] = prepend_search_paths(
        [library_dir],
        env.get("LIBRARY_PATH", ""),
    )
    return env, cccl


MINKOWSKI_BUILD_IDENTITY_PROBE = r"""
import json
import platform
import sys
import sysconfig
import torch

print(json.dumps({
    "python_version": list(sys.version_info[:3]),
    "python_cache_tag": sys.implementation.cache_tag,
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "system": platform.system().lower(),
    "machine": platform.machine().lower(),
    "torch_version": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
}, sort_keys=True, separators=(",", ":")))
"""


def probe_minkowski_build_identity(
    config: SetupConfig,
    lane: RuntimeLane,
    venv_python: Path,
) -> dict[str, Any]:
    process = run_command(
        config,
        [str(venv_python), "-B", "-c", MINKOWSKI_BUILD_IDENTITY_PROBE],
        step="minkowski-build-identity",
    )
    try:
        identity = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SetupContractError(
            "minkowski_build_identity_invalid",
            "Could not parse Python/Torch identity for the MinkowskiEngine wheel.",
            step="minkowski-build-identity",
        ) from exc
    if (
        not isinstance(identity, dict)
        or str(identity.get("torch_version", "")).split("+", 1)[0]
        != lane.torch_version
        or identity.get("torch_cuda") != lane.expected_torch_cuda
    ):
        raise SetupContractError(
            "minkowski_build_identity_mismatch",
            "Installed Torch does not match the selected MinkowskiEngine build lane.",
            step="minkowski-build-identity",
            details={"identity": identity, "lane": lane_payload(lane)},
        )
    return identity


def minkowski_cache_identity(
    config: SetupConfig,
    lane: RuntimeLane,
    toolkit: Mapping[str, Any],
    python_torch: Mapping[str, Any],
    openblas: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(openblas["artifact"])
    return {
        "repository": MINKOWSKI_REPOSITORY,
        "commit": MINKOWSKI_COMMIT,
        "patchset": MINKOWSKI_PATCHSET if lane.cuda_code >= 130 else "none",
        "lane_id": lane.lane_id,
        "cuda_code": lane.cuda_code,
        "cuda_release": str(toolkit["release"]),
        "cuda_home": str(Path(str(toolkit["cuda_home"])).resolve()),
        "gpu_sm": config.gpu_sm,
        "torch_cuda_arch_list": lane.torch_cuda_arch_list,
        "python_torch": dict(python_torch),
        "openblas": {
            "repository": OPENBLAS_REPOSITORY,
            "commit": OPENBLAS_COMMIT,
            "cache_key": openblas["cache_key"],
            "archive_sha256": artifact["archive_sha256"],
        },
    }


def validate_minkowski_wheel_cache(
    config: SetupConfig,
    cache_dir: Path,
    *,
    expected_cache_key: str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    require_path_inside(
        config.build_deps_dir,
        cache_dir,
        step="minkowski-cache-validation",
    )
    provenance = read_json_object(
        cache_dir / "provenance.json",
        code="minkowski_provenance_invalid",
        step="minkowski-cache-validation",
    )
    if (
        provenance.get("schema") != MINKOWSKI_CACHE_SCHEMA
        or provenance.get("cache_key") != expected_cache_key
        or provenance.get("identity") != dict(expected_identity)
    ):
        raise SetupContractError(
            "minkowski_provenance_mismatch",
            "MinkowskiEngine wheel provenance does not match the build fingerprint.",
            step="minkowski-cache-validation",
        )
    wheels = sorted(cache_dir.glob("*.whl"))
    wheel_record = provenance.get("wheel")
    if (
        len(wheels) != 1
        or not isinstance(wheel_record, dict)
        or wheel_record.get("filename") != wheels[0].name
        or wheels[0].is_symlink()
        or not wheels[0].is_file()
    ):
        raise SetupContractError(
            "minkowski_wheel_cache_invalid",
            "MinkowskiEngine cache must contain one regular provenance-matched wheel.",
            step="minkowski-cache-validation",
        )
    wheel = require_path_inside(
        cache_dir,
        wheels[0],
        step="minkowski-cache-validation",
    )
    actual_hash = file_sha256(wheel)
    actual_size = wheel.stat().st_size
    if (
        wheel_record.get("sha256") != actual_hash
        or wheel_record.get("size") != actual_size
    ):
        raise SetupContractError(
            "minkowski_wheel_hash_mismatch",
            "MinkowskiEngine cached wheel differs from its recorded hash or size.",
            step="minkowski-cache-validation",
        )
    return {
        "schema": MINKOWSKI_CACHE_SCHEMA,
        "cache_key": expected_cache_key,
        "cache_dir": str(cache_dir),
        "repository": MINKOWSKI_REPOSITORY,
        "commit": MINKOWSKI_COMMIT,
        "patch": dict(provenance.get("patch") or {}),
        "cccl_include": provenance.get("cccl_include"),
        "wheel": {
            "path": str(wheel),
            "filename": wheel.name,
            "sha256": actual_hash,
            "size": actual_size,
        },
        "identity": dict(expected_identity),
    }


def ensure_minkowski_wheel(
    config: SetupConfig,
    lane: RuntimeLane,
    venv_python: Path,
    toolkit: Mapping[str, Any],
    openblas: Mapping[str, Any],
    python_torch: Mapping[str, Any],
) -> dict[str, Any]:
    identity = minkowski_cache_identity(
        config,
        lane,
        toolkit,
        python_torch,
        openblas,
    )
    cache_key = canonical_sha256(identity)
    cache_dir = config.build_deps_dir / "minkowski" / cache_key
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        try:
            return validate_minkowski_wheel_cache(
                config,
                cache_dir,
                expected_cache_key=cache_key,
                expected_identity=identity,
            )
        except SetupContractError as exc:
            invalidate_cache_directory(
                config,
                cache_dir,
                step="minkowski-cache-validation",
                reason=str(exc),
            )

    staging = config.build_deps_dir / f".staging-minkowski-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    source = staging / "source"
    dist = staging / "dist"
    try:
        checkout_pinned_repository(
            config,
            repository=MINKOWSKI_REPOSITORY,
            commit=MINKOWSKI_COMMIT,
            destination=source,
            clone_step="minkowski-clone",
            checkout_step="minkowski-checkout",
        )
        patch = apply_minkowski_compatibility_patches(
            source,
            cuda_code=lane.cuda_code,
        )
        openblas_prefix = (
            Path(str(openblas["cache_dir"])) / "prefix"
        ).resolve()
        env, cccl = minkowski_build_environment(
            lane,
            toolkit,
            openblas_prefix,
        )
        dist.mkdir(parents=True, exist_ok=False)
        run_command(
            config,
            [
                str(venv_python),
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                str(dist),
            ],
            step="minkowski-wheel-build",
            env=env,
            cwd=source,
        )
        wheels = sorted(dist.glob("*.whl"))
        if (
            len(wheels) != 1
            or wheels[0].is_symlink()
            or not wheels[0].is_file()
            or not wheels[0].name.lower().startswith("minkowskiengine-")
        ):
            raise SetupContractError(
                "minkowski_wheel_build_invalid",
                "Pinned MinkowskiEngine build must produce exactly one regular wheel.",
                step="minkowski-wheel-build",
                details={"wheels": [item.name for item in wheels]},
            )
        cached_wheel = staging / wheels[0].name
        shutil.copy2(wheels[0], cached_wheel)
        wheel_record = {
            "filename": cached_wheel.name,
            "sha256": file_sha256(cached_wheel),
            "size": cached_wheel.stat().st_size,
        }
        remove_build_tree(config, source, step="minkowski-wheel-build")
        remove_build_tree(config, dist, step="minkowski-wheel-build")
        atomic_write_json(
            staging / "provenance.json",
            {
                "schema": MINKOWSKI_CACHE_SCHEMA,
                "cache_key": cache_key,
                "identity": identity,
                "patch": patch,
                "cccl_include": None if cccl is None else str(cccl),
                "wheel": wheel_record,
            },
        )
        promote_staging_directory(
            config,
            staging,
            cache_dir,
            step="minkowski-cache-promotion",
        )
    finally:
        if staging.exists():
            remove_build_tree(config, staging, step="minkowski-cache-cleanup")
    return validate_minkowski_wheel_cache(
        config,
        cache_dir,
        expected_cache_key=cache_key,
        expected_identity=identity,
    )


def install_dependencies(
    config: SetupConfig,
    lane: RuntimeLane,
    venv_python: Path,
    toolkit: Mapping[str, Any],
) -> dict[str, Any]:
    env = native_build_environment(lane, toolkit)
    pip_base = [
        str(venv_python), "-m", "pip", "install", "--no-cache-dir",
        "--retries", "5", "--timeout", "60",
    ]
    run_command(config, [*pip_base, "--upgrade", *PIP_BOOTSTRAP], step="pip-bootstrap", env=env)
    run_command(
        config,
        [
            *pip_base, "--index-url", lane.torch_index_url,
            f"torch=={lane.torch_version}",
            f"torchvision=={lane.torchvision_version}",
        ],
        step="torch-install",
        env=env,
    )
    run_command(
        config,
        [*pip_base, "--no-build-isolation", "-r", str(config.requirements_path)],
        step="python-dependencies-install",
        env=env,
    )
    openblas = ensure_openblas_dependency(config, lane)
    python_torch = probe_minkowski_build_identity(config, lane, venv_python)
    minkowski = ensure_minkowski_wheel(
        config,
        lane,
        venv_python,
        toolkit,
        openblas,
        python_torch,
    )
    wheel_path = Path(str(minkowski["wheel"]["path"])).resolve()
    require_path_inside(
        config.build_deps_dir,
        wheel_path,
        step="minkowskiengine-install",
    )
    run_command(
        config,
        [
            *pip_base,
            "--force-reinstall",
            "--no-deps",
            str(wheel_path),
        ],
        step="minkowskiengine-install",
        env=env,
    )
    return {
        "schema": "modly.gaussiangpt-build-dependencies.v1",
        "openblas": openblas,
        "minkowskiengine": minkowski,
        "install": {
            "source": "extension-local-cached-wheel",
            "wheel_path": str(wheel_path),
            "no_deps": True,
        },
    }


RUNTIME_PROBE = r"""
import importlib
import json
import sys
import traceback

result = {
    "ok": False, "python": sys.executable, "imports": {}, "torch": {},
    "cuda_tensor": {}, "minkowski_cuda": {}, "pytorch3d_cuda": {},
    "gsplat_cuda": {},
}
required = json.loads(sys.argv[1])
expected_torch = sys.argv[2]
expected_cuda = sys.argv[3]
expected_gpu_sm = None if sys.argv[4] == "" else int(sys.argv[4])
try:
    for name in required:
        try:
            importlib.import_module(name)
            result["imports"][name] = {"ok": True}
        except Exception as exc:
            result["imports"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            raise

    import torch
    result["torch"] = {
        "version": torch.__version__, "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    }
    if not str(torch.__version__).startswith(expected_torch):
        raise RuntimeError(f"torch version {torch.__version__} does not match {expected_torch}")
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(f"torch CUDA {torch.version.cuda} does not match {expected_cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(0)
    actual_sm = int(capability[0] * 10 + capability[1])
    result["torch"].update({
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(capability), "gpu_sm": actual_sm,
    })
    if expected_gpu_sm is not None and actual_sm != expected_gpu_sm:
        raise RuntimeError(f"CUDA device SM {actual_sm} does not match requested gpu_sm {expected_gpu_sm}")

    tensor = torch.tensor([2.0, 3.0], device=device)
    value = float((tensor * tensor).sum().item())
    torch.cuda.synchronize()
    if value != 13.0:
        raise RuntimeError(f"CUDA tensor smoke returned {value}, expected 13.0")
    result["cuda_tensor"] = {"ok": True, "value": value}

    import MinkowskiEngine as ME
    coordinates = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device)
    features = torch.ones((1, 1), dtype=torch.float32, device=device)
    sparse = ME.SparseTensor(features=features, coordinates=coordinates)
    sparse_value = float(sparse.F.sum().item())
    torch.cuda.synchronize()
    if sparse_value != 1.0:
        raise RuntimeError(f"MinkowskiEngine CUDA smoke returned {sparse_value}, expected 1.0")
    result["minkowski_cuda"] = {"ok": True, "value": sparse_value}

    from pytorch3d.transforms import quaternion_apply, quaternion_multiply
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    point = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    rotated = quaternion_apply(quaternion, point)
    multiplied = quaternion_multiply(quaternion, quaternion)
    torch.cuda.synchronize()
    if not torch.allclose(rotated, point):
        raise RuntimeError("PyTorch3D CUDA quaternion smoke changed identity rotation")
    if not torch.allclose(multiplied, quaternion):
        raise RuntimeError("PyTorch3D CUDA quaternion multiply changed identity quaternion")
    result["pytorch3d_cuda"] = {
        "ok": True,
        "applied": rotated.detach().cpu().tolist(),
        "multiplied": multiplied.detach().cpu().tolist(),
    }

    from gsplat import rasterization
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    opacities = torch.tensor([0.9], device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alphas, _meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=intrinsics,
        width=8, height=8, packed=False,
    )
    torch.cuda.synchronize()
    if not bool(torch.isfinite(rendered).all()) or not bool(torch.isfinite(alphas).all()):
        raise RuntimeError("gsplat CUDA rasterization produced non-finite output")
    result["gsplat_cuda"] = {
        "ok": True, "render_shape": list(rendered.shape),
        "alpha_shape": list(alphas.shape),
    }
    result["ok"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()

print(json.dumps(result, sort_keys=True, allow_nan=False))
raise SystemExit(0 if result["ok"] else 1)
"""


PIP_CHECK_PROBE = r"""
import json
import subprocess
import sys

completed = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=110,
    check=False,
)
print(json.dumps(
    {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    },
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
))
"""


CUSPARSELT_SBSA_PROBE = r"""
import base64
import csv
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import stat
import struct
import sys
from pathlib import Path

expected = json.loads(sys.argv[1])
result = {
    "schema": expected["schema"],
    "ok": False,
    "runtime": {},
    "package": {},
    "torch": {},
    "wheel": {},
    "library": {},
}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

try:
    runtime = {
        "implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "system": platform.system().lower(),
        "machine": platform.machine().lower(),
    }
    result["runtime"] = runtime
    require(runtime["implementation"] == "cpython", "target interpreter is not CPython")
    require(runtime["python_version"][:2] == [3, 11], "target interpreter is not CPython 3.11")
    require(runtime["system"] == "linux", "target platform is not Linux")
    require(runtime["machine"] == "aarch64", "target machine is not aarch64")

    distribution = metadata.distribution(expected["package_name"])
    package = {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
    }
    result["package"] = package
    require(package["name"] == expected["package_name"], "cuSPARSELt package name mismatch")
    require(package["version"] == expected["package_version"], "cuSPARSELt package version mismatch")

    torch_distribution = metadata.distribution("torch")
    torch_requirements = list(
        torch_distribution.metadata.get_all("Requires-Dist") or []
    )
    cusparselt_requirements = [
        requirement
        for requirement in torch_requirements
        if requirement.lower().startswith(expected["package_name"].lower())
    ]
    torch_result = {
        "version": torch_distribution.version,
        "cusparselt_requires_dist": cusparselt_requirements,
    }
    result["torch"] = torch_result
    require(
        torch_result["version"] == expected["torch_version"],
        "torch distribution version mismatch",
    )
    require(
        cusparselt_requirements == [expected["torch_requirement"]],
        "torch cuSPARSELt Requires-Dist is not the exact audited pin",
    )

    files = list(distribution.files or ())
    require(bool(files), "cuSPARSELt distribution has no file inventory")
    wheel_files = [item for item in files if item.name == "WHEEL"]
    require(len(wheel_files) == 1, "cuSPARSELt distribution must contain one WHEEL file")
    wheel_file = wheel_files[0]
    wheel_path = Path(distribution.locate_file(wheel_file))
    require(wheel_path.is_file(), "cuSPARSELt WHEEL file is missing")
    wheel_tags = [
        line.split(":", 1)[1].strip()
        for line in wheel_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Tag:")
    ]
    require(
        wheel_tags == [expected["wheel_tag"]],
        "cuSPARSELt WHEEL tag is not the singleton audited SBSA tag",
    )
    result["wheel"] = {
        "path": wheel_file.as_posix(),
        "tags": wheel_tags,
    }

    library_files = [
        item for item in files
        if item.as_posix() == expected["library_path"]
    ]
    require(
        len(library_files) == 1,
        "cuSPARSELt distribution must contain the exact library path once",
    )
    library_file = library_files[0]
    candidate = Path(distribution.locate_file(library_file))
    require(not candidate.is_symlink(), "cuSPARSELt library must not be a symlink")
    library_stat = candidate.lstat()
    require(stat.S_ISREG(library_stat.st_mode), "cuSPARSELt library must be a regular file")
    require(library_stat.st_size > 0, "cuSPARSELt library must be nonempty")

    site_root = Path(distribution.locate_file("")).resolve(strict=True)
    resolved_library = candidate.resolve(strict=True)
    contained_path = resolved_library.relative_to(site_root).as_posix()
    require(
        contained_path == expected["library_path"],
        "cuSPARSELt library does not resolve to the exact contained path",
    )

    record_files = [
        item for item in files
        if item.parent == wheel_file.parent and item.name == "RECORD"
    ]
    require(len(record_files) == 1, "cuSPARSELt distribution must contain one RECORD file")
    record_path = Path(distribution.locate_file(record_files[0]))
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        record_rows = list(csv.reader(handle, strict=True))
    require(
        all(len(row) == 3 for row in record_rows),
        "cuSPARSELt RECORD contains a malformed row",
    )
    library_rows = [
        row for row in record_rows if row[0] == expected["library_path"]
    ]
    require(
        len(library_rows) == 1,
        "cuSPARSELt RECORD must contain the exact library path once",
    )
    _record_path, record_hash, record_size_text = library_rows[0]
    algorithm, separator, recorded_digest = record_hash.partition("=")
    require(
        algorithm == "sha256" and separator == "=" and bool(recorded_digest),
        "cuSPARSELt RECORD must use a nonempty sha256 digest",
    )
    require(
        record_size_text.isdigit(),
        "cuSPARSELt RECORD library size must be an unsigned decimal integer",
    )
    record_size = int(record_size_text)
    require(
        record_size == library_stat.st_size,
        "cuSPARSELt RECORD size does not match the library",
    )
    with resolved_library.open("rb") as handle:
        computed_digest = base64.urlsafe_b64encode(
            hashlib.file_digest(handle, "sha256").digest()
        ).rstrip(b"=").decode("ascii")
    require(
        recorded_digest == computed_digest,
        "cuSPARSELt RECORD sha256 does not match the library",
    )

    with resolved_library.open("rb") as handle:
        header = handle.read(20)
    require(len(header) == 20, "cuSPARSELt library has a truncated ELF header")
    require(header[:4] == b"\x7fELF", "cuSPARSELt library is not ELF")
    elf = {
        "class": header[4],
        "data": header[5],
        "machine": struct.unpack("<H", header[18:20])[0],
    }
    require(elf["class"] == 2, "cuSPARSELt library is not ELF64")
    require(elf["data"] == 1, "cuSPARSELt library is not little-endian")
    require(elf["machine"] == 183, "cuSPARSELt library is not EM_AARCH64")

    mode = os.RTLD_NOW | os.RTLD_LOCAL
    loaded = ctypes.CDLL(str(resolved_library), mode=mode)
    loaded_symbols = []
    for symbol in expected["required_symbols"]:
        getattr(loaded, symbol)
        loaded_symbols.append(symbol)

    get_property = loaded.cusparseLtGetProperty
    get_property.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    get_property.restype = ctypes.c_int
    properties = []
    for enum, expected_value in enumerate(expected["property_values"]):
        value = ctypes.c_int(-1)
        status = int(get_property(enum, ctypes.byref(value)))
        properties.append({
            "enum": enum,
            "status": status,
            "value": int(value.value),
        })
        require(status == 0, f"cusparseLtGetProperty({enum}) returned {status}")
        require(
            value.value == expected_value,
            f"cusparseLtGetProperty({enum}) returned {value.value}",
        )

    result["library"] = {
        "relative_path": expected["library_path"],
        "resolved_path": str(resolved_library),
        "size": int(library_stat.st_size),
        "record": {
            "sha256": recorded_digest,
            "size": record_size,
        },
        "elf": elf,
        "load": {
            "mode": int(mode),
            "flags": ["RTLD_NOW", "RTLD_LOCAL"],
            "symbols": loaded_symbols,
        },
        "properties": properties,
        "version": ".".join(str(item) for item in expected["property_values"]),
    }
    result["ok"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(
    result,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
))
raise SystemExit(0 if result["ok"] else 1)
"""


def is_exact_cusparselt_pip_check_exception(
    lane: RuntimeLane,
    returncode: int,
    stdout: str,
    stderr: str,
) -> bool:
    return (
        returncode == 1
        and stdout == CUSPARSELT_PIP_CHECK_OUTPUT
        and stderr == ""
        and lane.lane_id == "linux-aarch64-cp311-cu130-experimental"
        and lane.system == "linux"
        and lane.machine == "aarch64"
        and lane.cuda_code == 130
        and lane.python_version == (3, 11)
        and lane.torch_version == "2.9.1"
        and lane.torchvision_version == "0.24.1"
        and lane.torch_index_url == "https://download.pytorch.org/whl/cu130"
        and lane.expected_torch_cuda == "13.0"
    )


def parse_canonical_probe_json(output: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(output, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError) as exc:
        raise SetupContractError(
            "cusparselt_probe_invalid",
            "cuSPARSELt SBSA probe did not emit one canonical JSON document.",
            step="pip-check",
            details={"output": output},
        ) from exc
    if not isinstance(payload, dict):
        raise SetupContractError(
            "cusparselt_probe_invalid",
            "cuSPARSELt SBSA probe JSON must be an object.",
            step="pip-check",
            details={"output": output},
        )
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise SetupContractError(
            "cusparselt_probe_invalid",
            "cuSPARSELt SBSA probe JSON is not canonicalizable.",
            step="pip-check",
            details={"validation": payload},
        ) from exc
    if output != canonical:
        raise SetupContractError(
            "cusparselt_probe_invalid",
            "cuSPARSELt SBSA probe output was not exact canonical JSON.",
            step="pip-check",
            details={"output": output},
        )
    return payload


def run_pip_check_probe(
    config: SetupConfig,
    venv_python: Path,
) -> dict[str, Any]:
    process = run_command(
        config,
        [str(venv_python), "-B", "-c", PIP_CHECK_PROBE],
        step="pip-check",
        check=False,
    )
    payload = parse_canonical_probe_json(process.stdout)
    if process.returncode != 0:
        raise SetupContractError(
            "pip_check_probe_failed",
            "The isolated pip-check capture process failed.",
            step="pip-check",
            details={
                "capture_returncode": process.returncode,
                "capture": payload,
            },
        )
    if set(payload) != {"returncode", "stdout", "stderr"}:
        raise SetupContractError(
            "pip_check_probe_invalid",
            "The isolated pip-check capture has an unexpected schema.",
            step="pip-check",
            details={"capture": payload},
        )
    if (
        type(payload.get("returncode")) is not int
        or not isinstance(payload.get("stdout"), str)
        or not isinstance(payload.get("stderr"), str)
    ):
        raise SetupContractError(
            "pip_check_probe_invalid",
            "The isolated pip-check capture has invalid field types.",
            step="pip-check",
            details={"capture": payload},
        )
    return payload


def validate_cusparselt_probe_payload(payload: Mapping[str, Any]) -> None:
    def reject(message: str) -> None:
        raise SetupContractError(
            "cusparselt_probe_invalid",
            message,
            step="pip-check",
            details={"validation": dict(payload)},
        )

    if set(payload) != {
        "schema", "ok", "runtime", "package", "torch", "wheel", "library"
    }:
        reject("cuSPARSELt success probe has an unexpected top-level schema.")
    if payload.get("schema") != CUSPARSELT_PROBE_SCHEMA:
        reject("cuSPARSELt success probe schema identifier is invalid.")
    if payload.get("ok") is not True:
        reject("cuSPARSELt success probe did not report ok=true.")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "implementation", "python_version", "system", "machine"
    }:
        reject("cuSPARSELt success probe runtime evidence is malformed.")
    python_version = runtime.get("python_version")
    if (
        runtime.get("implementation") != "cpython"
        or runtime.get("system") != "linux"
        or runtime.get("machine") != "aarch64"
        or not isinstance(python_version, list)
        or len(python_version) != 3
        or any(type(item) is not int for item in python_version)
        or python_version[:2] != [3, 11]
    ):
        reject("cuSPARSELt success probe runtime is not Linux aarch64 CPython 3.11.")

    if payload.get("package") != {
        "name": "nvidia-cusparselt-cu13",
        "version": "0.8.0",
    }:
        reject("cuSPARSELt success probe package identity is invalid.")
    if payload.get("torch") != {
        "version": CUSPARSELT_TORCH_DISTRIBUTION_VERSION,
        "cusparselt_requires_dist": [CUSPARSELT_TORCH_REQUIREMENT],
    }:
        reject("cuSPARSELt success probe torch metadata is invalid.")

    wheel = payload.get("wheel")
    if (
        not isinstance(wheel, dict)
        or set(wheel) != {"path", "tags"}
        or not isinstance(wheel.get("path"), str)
        or not wheel["path"].endswith(".dist-info/WHEEL")
        or wheel.get("tags") != [CUSPARSELT_WHEEL_TAG]
    ):
        reject("cuSPARSELt success probe WHEEL evidence is invalid.")

    library = payload.get("library")
    if not isinstance(library, dict) or set(library) != {
        "relative_path", "resolved_path", "size", "record", "elf",
        "load", "properties", "version",
    }:
        reject("cuSPARSELt success probe library evidence is malformed.")
    if library.get("relative_path") != CUSPARSELT_LIBRARY_PATH:
        reject("cuSPARSELt success probe library relative path is invalid.")
    resolved_path = library.get("resolved_path")
    if (
        not isinstance(resolved_path, str)
        or not Path(resolved_path).is_absolute()
        or not resolved_path.endswith("/" + CUSPARSELT_LIBRARY_PATH)
    ):
        reject("cuSPARSELt success probe resolved library path is invalid.")
    size = library.get("size")
    if type(size) is not int or size <= 0:
        reject("cuSPARSELt success probe library size is invalid.")
    record = library.get("record")
    if (
        not isinstance(record, dict)
        or set(record) != {"sha256", "size"}
        or record.get("size") != size
        or not isinstance(record.get("sha256"), str)
        or re.fullmatch(r"[A-Za-z0-9_-]{43}", record["sha256"]) is None
    ):
        reject("cuSPARSELt success probe RECORD evidence is invalid.")
    if library.get("elf") != {"class": 2, "data": 1, "machine": 183}:
        reject("cuSPARSELt success probe ELF evidence is invalid.")
    load = library.get("load")
    if (
        not isinstance(load, dict)
        or set(load) != {"mode", "flags", "symbols"}
        or type(load.get("mode")) is not int
        or load["mode"] <= 0
        or load.get("flags") != ["RTLD_NOW", "RTLD_LOCAL"]
        or load.get("symbols") != list(CUSPARSELT_REQUIRED_SYMBOLS)
    ):
        reject("cuSPARSELt success probe dynamic-load evidence is invalid.")
    expected_properties = [
        {"enum": 0, "status": 0, "value": 0},
        {"enum": 1, "status": 0, "value": 8},
        {"enum": 2, "status": 0, "value": 0},
    ]
    if (
        library.get("properties") != expected_properties
        or library.get("version") != "0.8.0"
    ):
        reject("cuSPARSELt success probe property/version evidence is invalid.")


def run_cusparselt_sbsa_probe(
    config: SetupConfig,
    venv_python: Path,
) -> dict[str, Any]:
    expectation = {
        "schema": CUSPARSELT_PROBE_SCHEMA,
        "package_name": "nvidia-cusparselt-cu13",
        "package_version": "0.8.0",
        "torch_version": CUSPARSELT_TORCH_DISTRIBUTION_VERSION,
        "torch_requirement": CUSPARSELT_TORCH_REQUIREMENT,
        "wheel_tag": CUSPARSELT_WHEEL_TAG,
        "library_path": CUSPARSELT_LIBRARY_PATH,
        "required_symbols": list(CUSPARSELT_REQUIRED_SYMBOLS),
        "property_values": [0, 8, 0],
    }
    process = run_command(
        config,
        [
            str(venv_python),
            "-B",
            "-c",
            CUSPARSELT_SBSA_PROBE,
            json.dumps(
                expectation,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        ],
        step="cusparselt-sbsa-probe",
        check=False,
    )
    payload = parse_canonical_probe_json(process.stdout)
    if process.returncode != 0 or payload.get("ok") is not True:
        raise SetupContractError(
            "cusparselt_probe_failed",
            "cuSPARSELt SBSA compatibility probe failed.",
            step="pip-check",
            details={
                "returncode": process.returncode,
                "validation": payload,
            },
        )
    validate_cusparselt_probe_payload(payload)
    return payload


def run_runtime_probe(config: SetupConfig, lane: RuntimeLane, venv_python: Path) -> dict[str, Any]:
    process = run_command(
        config,
        [
            str(venv_python), "-B", "-c", RUNTIME_PROBE,
            json.dumps(REQUIRED_RUNTIME_IMPORTS), lane.torch_version,
            lane.expected_torch_cuda,
            "" if config.gpu_sm is None else str(config.gpu_sm),
        ],
        step="runtime-native-cuda-probe",
        check=False,
    )
    output_lines = [line for line in process.stdout.splitlines() if line.strip()]
    try:
        probe = json.loads(output_lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SetupContractError("runtime_probe_invalid", f"Runtime probe did not emit valid JSON. See {config.log_path}.", step="runtime-native-cuda-probe") from exc
    if process.returncode != 0 or probe.get("ok") is not True:
        raise SetupContractError(
            "runtime_probe_failed",
            "GaussianGPT native/CUDA runtime probe failed: "
            + str(probe.get("error") or f"exit {process.returncode}")
            + f". See {config.log_path}.",
            step="runtime-native-cuda-probe",
            details={"probe": probe},
        )
    pip_check = run_pip_check_probe(config, venv_python)
    pip_evidence: dict[str, Any] = {
        "ok": pip_check["returncode"] == 0 and pip_check["stderr"] == "",
        "pip_check_passed": (
            pip_check["returncode"] == 0 and pip_check["stderr"] == ""
        ),
        "returncode": pip_check["returncode"],
        "stdout": pip_check["stdout"],
        "stderr": pip_check["stderr"],
        "normalized_exception": None,
    }
    probe["pip_check"] = pip_evidence
    if pip_evidence["pip_check_passed"]:
        return probe

    if not is_exact_cusparselt_pip_check_exception(
        lane,
        pip_check["returncode"],
        pip_check["stdout"],
        pip_check["stderr"],
    ):
        raise SetupContractError(
            "pip_check_failed",
            f"pip check failed without the exact audited SBSA exception. See {config.log_path}.",
            step="pip-check",
            details={"probe": probe},
        )

    try:
        validation = run_cusparselt_sbsa_probe(config, venv_python)
    except SetupContractError as exc:
        pip_evidence["normalized_exception"] = {
            "applied": False,
            "kind": "validated-cu130-linux-aarch64-sbsa-wheel",
            "original_returncode": pip_check["returncode"],
            "original_stdout": pip_check["stdout"],
            "original_stderr": pip_check["stderr"],
            "validation_failure": {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
            },
        }
        raise SetupContractError(
            "pip_check_failed",
            f"pip check SBSA exception validation failed. See {config.log_path}.",
            step="pip-check",
            details={"probe": probe},
        ) from exc

    pip_evidence.update({
        "ok": True,
        "pip_check_passed": False,
        "normalized_exception": {
            "applied": True,
            "kind": "validated-cu130-linux-aarch64-sbsa-wheel",
            "original_returncode": pip_check["returncode"],
            "original_stdout": pip_check["stdout"],
            "original_stderr": pip_check["stderr"],
            "validation": validation,
        },
    })
    return probe


def lane_payload(lane: RuntimeLane | None) -> dict[str, Any] | None:
    if lane is None:
        return None
    value = asdict(lane)
    value["python_version"] = ".".join(str(item) for item in lane.python_version)
    value["risks"] = list(lane.risks)
    return value


def write_status(
    config: SetupConfig,
    *,
    status: str,
    ready: bool,
    code: str,
    message: str,
    lane: RuntimeLane | None,
    toolkit: Mapping[str, Any] | None = None,
    anchor: Mapping[str, Any] | None = None,
    probe: Mapping[str, Any] | None = None,
    build_dependencies: Mapping[str, Any] | None = None,
    failure: SetupContractError | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema": "modly.setup-status.v1",
        "extension_id": EXTENSION_ID,
        "status": status,
        "ready": ready,
        "code": code,
        "message": message,
        "checked_at": utc_now(),
        "validate_only": config.validate_only,
        "no_install": config.no_install,
        "ext_dir": str(config.ext_dir),
        "venv_python": str(config.venv_python),
        "venv_python_present": config.venv_python.is_file(),
        "downloads": {
            "weights": False, "datasets": False, "gaussiangpt_source": False,
            "dependencies_only_during_install": not (config.validate_only or config.no_install),
        },
        "upstream_commit": UPSTREAM_COMMIT,
        "driver_cuda_hint": config.cuda_version,
        "actual_cuda_toolkit": dict(toolkit or {}),
        "lane": lane_payload(lane),
        "runtime_build_risks": [MINKOWSKI_BUILD_RISK],
        "build_dependency_plan": {
            "root": str(config.build_deps_dir),
            "openblas": {
                "repository": OPENBLAS_REPOSITORY,
                "commit": OPENBLAS_COMMIT,
                "build_arguments": list(OPENBLAS_BUILD_ARGUMENTS),
                "static_only": True,
            },
            "minkowskiengine": {
                "repository": MINKOWSKI_REPOSITORY,
                "commit": MINKOWSKI_COMMIT,
                "cuda13_patchset": MINKOWSKI_PATCHSET,
                "install_source": "extension-local-cached-wheel",
            },
        },
        "build_dependencies": dict(build_dependencies or {}),
        "anchor": dict(anchor or {}),
        "probe": dict(probe or {}),
        "log_path": str(config.log_path),
    }
    if failure is not None:
        payload["failure"] = {
            "code": failure.code, "step": failure.step,
            "message": str(failure), "details": failure.details,
        }
    config.status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.status_path.with_name(f".{config.status_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, config.status_path)
    finally:
        temporary.unlink(missing_ok=True)
    return config.status_path


def execute(config: SetupConfig) -> int:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    append_log(config, "setup-start " + json.dumps({
        "validate_only": config.validate_only, "no_install": config.no_install,
        "ext_dir": str(config.ext_dir), "gpu_sm": config.gpu_sm,
        "cuda_version": config.cuda_version,
    }, sort_keys=True))
    anchor: dict[str, Any] | None = None
    lane: RuntimeLane | None = None
    toolkit: dict[str, Any] | None = None
    build_dependencies: dict[str, Any] = {}
    try:
        anchor = validate_extension_anchor(config)
        python_exe = resolve_python(config.python_exe)
        python_version = probe_python_version(config, python_exe)
        toolkits = probe_cuda_toolkits(config)
        lane, toolkit = select_lane(config, python_version, toolkits)
        append_log(config, f"lane-selected: {lane.lane_id}")
        append_log(
            config,
            f"toolkit-selected: CUDA {toolkit['release']} at {toolkit['cuda_home']}; "
            f"driver-hint={config.cuda_version!r}",
        )
        append_log(config, f"lane-risk: {MINKOWSKI_BUILD_RISK}")

        if config.validate_only:
            if not config.venv_python.is_file():
                message = (
                    "Validation completed: extension metadata is valid, but the "
                    "extension-local venv is absent. No files were installed."
                )
                write_status(
                    config, status="blocked", ready=False, code="venv_missing",
                    message=message, lane=lane, anchor=anchor,
                    toolkit=toolkit,
                    probe={"status": "not_run", "reason": "venv_missing"},
                )
                emit(message)
                return 0
            try:
                probe = run_runtime_probe(config, lane, config.venv_python)
            except SetupContractError as exc:
                write_status(
                    config, status="blocked", ready=False, code=exc.code,
                    message=str(exc), lane=lane, anchor=anchor,
                    toolkit=toolkit,
                    probe=exc.details.get("probe", {}), failure=exc,
                )
                emit(f"Validation completed with blocked readiness: {exc}")
                return 0
            message = "Validation completed: extension-local runtime passed all probes."
            write_status(
                config, status="ready", ready=True, code="ready",
                message=message, lane=lane, anchor=anchor, probe=probe,
                toolkit=toolkit,
            )
            emit(message)
            return 0

        if config.no_install:
            if not config.venv_python.is_file():
                raise SetupContractError(
                    "venv_missing",
                    f"--no-install requires an existing {config.venv_python}.",
                    step="no-install-validation",
                )
            probe = run_runtime_probe(config, lane, config.venv_python)
            message = "Existing extension-local runtime passed all probes."
            write_status(
                config, status="ready", ready=True, code="ready",
                message=message, lane=lane, anchor=anchor, probe=probe,
                toolkit=toolkit,
            )
            emit(message)
            return 0

        venv_python = create_extension_venv(config, python_exe)
        build_dependencies = install_dependencies(
            config,
            lane,
            venv_python,
            toolkit,
        )
        probe = run_runtime_probe(config, lane, venv_python)
        message = (
            "GaussianGPT dependency installation completed and all strict "
            "native/CUDA probes passed. Model weights remain managed by Modly."
        )
        write_status(
            config, status="ready", ready=True, code="ready",
            message=message, lane=lane, anchor=anchor, probe=probe,
            toolkit=toolkit, build_dependencies=build_dependencies,
        )
        emit(message)
        return 0
    except SetupContractError as exc:
        write_status(
            config, status="blocked", ready=False, code=exc.code,
            message=str(exc), lane=lane, anchor=anchor,
            toolkit=toolkit, build_dependencies=build_dependencies,
            probe=exc.details.get("probe", {}), failure=exc,
        )
        append_log(config, f"setup-blocked code={exc.code} step={exc.step}: {exc}")
        emit(f"FAILED [{exc.code}] at {exc.step}: {exc} Read {config.status_path} and {config.log_path}.")
        return 1
    except Exception as exc:
        wrapped = SetupContractError(
            "unexpected_setup_failure", f"{type(exc).__name__}: {exc}",
            step="unexpected",
        )
        write_status(
            config, status="blocked", ready=False, code=wrapped.code,
            message=str(wrapped), lane=lane, anchor=anchor, failure=wrapped,
            toolkit=toolkit, build_dependencies=build_dependencies,
        )
        append_log(config, f"setup-unexpected: {type(exc).__name__}: {exc}")
        emit(f"FAILED [{wrapped.code}]: {wrapped}. Read {config.status_path} and {config.log_path}.")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        config = parse_setup_config(raw_args)
    except SetupContractError as exc:
        emit(f"FAILED [{exc.code}] at {exc.step}: {exc}")
        return 2
    return execute(config)


if __name__ == "__main__":
    raise SystemExit(main())
