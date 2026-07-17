"""Modly runtime adapter for the vendored GaussianGPT implementation.

The extension performs no network access. Model assets are owned by Modly's
HTTPS downloader and upstream source is vendored at a pinned commit.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except ModuleNotFoundError:  # pragma: no cover - standalone tests outside Modly
    class GenerationCancelled(Exception):
        """Fallback cancellation exception used outside Modly."""

    class BaseGenerator:  # type: ignore[override]
        MODEL_ID = ""
        DISPLAY_NAME = ""
        VRAM_GB = 0

        def __init__(self, model_dir: Path | str, outputs_dir: Path | str) -> None:
            self.model_dir = Path(model_dir)
            self.outputs_dir = Path(outputs_dir)
            self.hf_repo = ""
            self.hf_downloads: list[dict[str, Any]] = []
            self.hf_skip_prefixes: list[str] = []
            self.download_check = ""
            self._params_schema: list[dict[str, Any]] = []


EXTENSION_ID = "gaussiangpt"
DEFAULT_NODE_ID = "generate-vfront"
MODEL_ID = f"{EXTENSION_ID}/{DEFAULT_NODE_ID}"
DISPLAY_NAME = "GaussianGPT 3D-FRONT"
DOWNLOAD_CHECK = ".modly/https-assets-ready.json"
UPSTREAM_COMMIT = "0615470a5ff359c676408e9ae42036d534d04a43"

EXTENSION_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EXTENSION_DIR / "manifest.json"
VENDOR_ROOT = EXTENSION_DIR / "vendor" / "GaussianGPT"
SETUP_STATUS_PATH = EXTENSION_DIR / ".modly" / "setup" / "setup-status.json"

MARKER_SCHEMA_VERSION = 1
MARKER_KIND = "modly.https-assets.ready"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
SH_C0 = 0.28209479177387814
DEPENDENCY_IMPORT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    display_name: str
    pair_name: str
    vqvae_filename: str
    gpt_filename: str
    https_downloads: tuple[Mapping[str, Any], Mapping[str, Any]]

    @property
    def model_id(self) -> str:
        return f"{EXTENSION_ID}/{self.node_id}"


NODE_CONFIGS: dict[str, NodeConfig] = {
    "generate-vfront": NodeConfig(
        node_id="generate-vfront",
        display_name="GaussianGPT 3D-FRONT",
        pair_name="vfront",
        vqvae_filename="vqvae_vfront.ckpt",
        gpt_filename="gpt_vfront.ckpt",
        https_downloads=(
            {
                "url": "https://kaldir.vc.cit.tum.de/gaussiangpt/vqvae_vfront.ckpt",
                "filename": "vqvae_vfront.ckpt",
                "size_bytes": 2115020643,
                "sha256": "9f70d0939dc791292be52da6c503bf51b3ac73d9905b51784d0aac81e44faf7a",
            },
            {
                "url": "https://kaldir.vc.cit.tum.de/gaussiangpt/gpt_vfront.ckpt",
                "filename": "gpt_vfront.ckpt",
                "size_bytes": 3421157765,
                "sha256": "203dc730495bf4f21e60280c6152703867f1b81e9c035d110d792e6a87d9313b",
            },
        ),
    ),
    "generate-both": NodeConfig(
        node_id="generate-both",
        display_name="GaussianGPT 3D-FRONT + ASE",
        pair_name="both",
        vqvae_filename="vqvae_both.ckpt",
        gpt_filename="gpt_both.ckpt",
        https_downloads=(
            {
                "url": "https://kaldir.vc.cit.tum.de/gaussiangpt/vqvae_both.ckpt",
                "filename": "vqvae_both.ckpt",
                "size_bytes": 2115018167,
                "sha256": "a780ed2920e877736699bb3da84a43362c1a565adff91217841d4b95cb784542",
            },
            {
                "url": "https://kaldir.vc.cit.tum.de/gaussiangpt/gpt_both.ckpt",
                "filename": "gpt_both.ckpt",
                "size_bytes": 3421156679,
                "sha256": "054978e811716292472dd501cc05b54c39789af6dd8730aab86061545c8f0a8b",
            },
        ),
    ),
}

DEPENDENCY_IMPORTS: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("lightning", "lightning"),
    ("omegaconf", "omegaconf"),
    ("hydra-core", "hydra"),
    ("MinkowskiEngine", "MinkowskiEngine"),
    ("vector-quantize-pytorch", "vector_quantize_pytorch"),
    ("numpy", "numpy"),
    ("plyfile", "plyfile"),
    ("einops", "einops"),
    ("imageio", "imageio"),
    ("gsplat", "gsplat"),
    ("torchvision", "torchvision"),
    ("torchmetrics", "torchmetrics"),
    ("lpips", "lpips"),
    ("pytorch3d", "pytorch3d"),
    ("Pillow", "PIL"),
)

REQUIRED_VENDOR_FILES: tuple[Path, ...] = (
    Path("LICENSE"),
    Path("model") / "gaussian_gpt.py",
    Path("model") / "gaussian_vqvae.py",
    Path("model") / "gpt" / "gpt.py",
    Path("model") / "vqvae" / "cnn" / "configurable_vqvae.py",
    Path("utils") / "render.py",
    Path("utils") / "config.py",
    Path("data") / "photoshape.py",
    Path("serialization") / "__init__.py",
    Path("conf") / "dataclasses.py",
)

ALLOWED_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "background_color",
        "render_preview",
    }
)

HOST_COMPAT_PARAMS = frozenset(
    {
        "remesh",
        "enable_texture",
        "texture_resolution",
    }
)


class AssetVerificationError(RuntimeError):
    """Raised when Modly-managed checkpoint readiness evidence is invalid."""


class SetupReadinessError(RuntimeError):
    """Raised when dependency setup evidence is missing, stale, or failed."""


def _log(message: str) -> None:
    print(f"[gaussiangpt] {message}", file=sys.stderr, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    progress_cb: Callable[..., Any] | None,
    percent: int,
    message: str,
) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(percent, message)
    except TypeError:
        progress_cb({"progress": percent, "phase": message})
    except Exception as exc:  # progress reporting must not corrupt a completed run
        _log(f"Progress callback failed and was ignored: {type(exc).__name__}: {exc}")


def _raise_if_cancelled(cancel_event: Any | None) -> None:
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        raise GenerationCancelled()


def _load_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read extension manifest at {MANIFEST_PATH}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Extension manifest is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Extension manifest root must be a JSON object.")
    return value


def _manifest_node(node_id: str) -> dict[str, Any]:
    nodes = _load_manifest().get("nodes")
    if not isinstance(nodes, list):
        raise RuntimeError("Extension manifest must contain a nodes array.")
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            return node
    raise RuntimeError(f"Extension manifest does not define node {node_id!r}.")


def _schema_for_node(node_id: str) -> list[dict[str, Any]]:
    node = _manifest_node(node_id)
    schema = node.get("params_schema")
    if not isinstance(schema, list):
        raise RuntimeError(f"Manifest node {node_id!r} has no params_schema array.")
    return [dict(item) for item in schema if isinstance(item, dict)]


DEPENDENCY_IMPORT_PROBE = r"""
import importlib
import json
import sys
import traceback

requirements = json.loads(sys.argv[1])
diagnostics = {}
for package, module_name in requirements:
    try:
        module = importlib.import_module(module_name)
        diagnostics[package] = {
            "ok": True,
            "module": module_name,
            "version": getattr(module, "__version__", None),
            "origin": getattr(module, "__file__", None),
        }
    except BaseException as exc:
        diagnostics[package] = {
            "ok": False,
            "module": module_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
print(json.dumps(diagnostics, sort_keys=True, allow_nan=False))
"""


def _dependency_status() -> dict[str, dict[str, Any]]:
    """Attempt real imports in a bounded child process."""

    command = [
        sys.executable,
        "-B",
        "-c",
        DEPENDENCY_IMPORT_PROBE,
        json.dumps(list(DEPENDENCY_IMPORTS)),
    ]
    try:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=DEPENDENCY_IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            package: {
                "ok": False,
                "module": module,
                "error": (
                    "Import probe exceeded "
                    f"{DEPENDENCY_IMPORT_TIMEOUT_SECONDS} seconds."
                ),
            }
            for package, module in DEPENDENCY_IMPORTS
        }
    except OSError as exc:
        return {
            package: {
                "ok": False,
                "module": module,
                "error": f"Import probe could not start: {type(exc).__name__}: {exc}",
            }
            for package, module in DEPENDENCY_IMPORTS
        }

    lines = [line for line in process.stdout.splitlines() if line.strip()]
    try:
        value = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        error = (
            f"Import probe emitted invalid diagnostics (exit {process.returncode}); "
            f"stderr={process.stderr.strip()!r}."
        )
        return {
            package: {"ok": False, "module": module, "error": error}
            for package, module in DEPENDENCY_IMPORTS
        }
    if not isinstance(value, dict):
        return {
            package: {
                "ok": False,
                "module": module,
                "error": "Import probe JSON root was not an object.",
            }
            for package, module in DEPENDENCY_IMPORTS
        }
    diagnostics: dict[str, dict[str, Any]] = {}
    for package, module in DEPENDENCY_IMPORTS:
        item = value.get(package)
        if not isinstance(item, dict):
            item = {
                "ok": False,
                "module": module,
                "error": "Import probe omitted this dependency.",
            }
        diagnostics[package] = dict(item)
    return diagnostics


def _vendor_status() -> dict[str, bool]:
    return {str(path): (VENDOR_ROOT / path).is_file() for path in REQUIRED_VENDOR_FILES}


def _canonical_plan_json(plan: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        list(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _plan_sha256(plan: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_plan_json(plan).encode("utf-8")).hexdigest()


def _expected_marker_assets(plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "filename": item["filename"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in plan
    ]


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssetVerificationError(f"Cannot read readiness marker {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssetVerificationError(f"Readiness marker is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AssetVerificationError("Readiness marker root must be a JSON object.")
    return value


def _sha256_file(
    path: Path,
    cancel_event: Any | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _raise_if_cancelled(cancel_event)
            block = handle.read(HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_float(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {parsed}.")
    return parsed


def _strict_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {limit}, got {value}.")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{name} must be true or false.")


def _validate_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ValueError("GaussianGPT params must be an object.")

    model_params = dict(params)
    host_compat_params = {
        name: model_params.pop(name)
        for name in HOST_COMPAT_PARAMS
        if name in model_params
    }

    if "remesh" in host_compat_params:
        remesh = host_compat_params["remesh"]
        if not isinstance(remesh, str) or remesh not in {"quad", "triangle", "none"}:
            raise ValueError("remesh must be one of: quad, triangle, none.")

    if "enable_texture" in host_compat_params:
        enable_texture = host_compat_params["enable_texture"]
        if not isinstance(enable_texture, bool):
            raise ValueError("enable_texture must be a boolean.")

    if "texture_resolution" in host_compat_params:
        texture_resolution = host_compat_params["texture_resolution"]
        if (
            isinstance(texture_resolution, bool)
            or not isinstance(texture_resolution, int)
            or texture_resolution <= 0
        ):
            raise ValueError("texture_resolution must be a positive integer.")

    unknown = sorted(str(key) for key in model_params if key not in ALLOWED_PARAMS)
    if unknown:
        raise ValueError(
            "Unsupported GaussianGPT parameter(s): "
            + ", ".join(unknown)
            + ". This unconditional extension does not accept prompt, image, model_variant, "
            "num_samples, max_length, or output-name controls."
        )

    temperature = _strict_float(
        model_params.get("temperature", 0.9),
        "temperature",
        minimum=0.0,
        maximum=2.0,
    )
    top_p = _strict_float(
        model_params.get("top_p", 0.9),
        "top_p",
        minimum=0.01,
        maximum=1.0,
    )
    top_k = _strict_int(model_params.get("top_k", 0), "top_k", minimum=0)
    seed = _strict_int(
        model_params.get("seed", 0),
        "seed",
        minimum=0,
        maximum=2147483646,
    )

    background_color = model_params.get("background_color", "white")
    if background_color not in {"white", "black"}:
        raise ValueError("background_color must be either 'white' or 'black'.")

    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "background_color": background_color,
        "render_preview": _strict_bool(
            model_params.get("render_preview", "true"),
            "render_preview",
        ),
        "num_samples": 1,
    }


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _canonical_scene_payload(scene: Any) -> dict[str, Any]:
    if not hasattr(scene, "to_dict"):
        raise TypeError("GaussianGPT.sample() returned a scene without to_dict().")
    raw = scene.to_dict()
    if not isinstance(raw, Mapping):
        raise TypeError("Gaussian scene to_dict() must return a mapping.")

    required = ("coords", "sh0", "opacities", "scales", "quats")
    missing = [key for key in required if key not in raw]
    if missing:
        raise RuntimeError("Gaussian scene omitted required fields: " + ", ".join(missing))

    payload: dict[str, Any] = {}
    for key in (*required, "sh"):
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        payload[key] = value

    _validate_scene_payload(payload)
    return payload


def _validate_scene_payload(payload: Mapping[str, Any]) -> int:
    import numpy as np

    def array(name: str) -> Any:
        try:
            result = np.asarray(_tensor_to_numpy(payload[name]))
        except Exception as exc:
            raise RuntimeError(f"Gaussian field {name!r} cannot be converted to an array.") from exc
        try:
            finite = np.isfinite(result)
        except TypeError as exc:
            raise RuntimeError(f"Gaussian field {name!r} is not numeric.") from exc
        if not bool(finite.all()):
            raise RuntimeError(f"Gaussian field {name!r} contains NaN or infinite values.")
        return result

    coords = array("coords")
    sh0 = array("sh0")
    opacities = array("opacities")
    scales = array("scales")
    quats = array("quats")

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"coords must have shape (N, 3), got {coords.shape}.")
    count = int(coords.shape[0])
    if count == 0:
        raise RuntimeError("GaussianGPT returned an empty Gaussian scene.")
    if sh0.shape != (count, 3):
        raise RuntimeError(f"sh0 must have shape ({count}, 3), got {sh0.shape}.")
    if opacities.shape not in {(count,), (count, 1)}:
        raise RuntimeError(
            f"opacities must have shape ({count},) or ({count}, 1), got {opacities.shape}."
        )
    if scales.shape != (count, 3):
        raise RuntimeError(f"scales must have shape ({count}, 3), got {scales.shape}.")
    if quats.shape != (count, 4):
        raise RuntimeError(f"quats must have shape ({count}, 4), got {quats.shape}.")

    if "sh" in payload:
        sh = array("sh")
        if sh.ndim != 2 or sh.shape[0] != count or sh.shape[1] % 3 != 0:
            raise RuntimeError(
                f"sh must have shape (N, D) with D divisible by 3, got {sh.shape}."
            )
    return count


def _align4(data: bytes, pad: bytes = b"\x00") -> bytes:
    remainder = len(data) % 4
    return data if remainder == 0 else data + pad * (4 - remainder)


def _write_point_cloud_glb(coords: Any, sh0: Any, path: Path) -> None:
    """Write a deterministic GLB POINTS preview from Gaussian centers/DC color.

    This intentionally does not encode Gaussian scale, rotation, opacity, or
    covariance and therefore is not a Gaussian-splat or mesh-fidelity artifact.
    """

    import numpy as np

    source_positions = np.asarray(_tensor_to_numpy(coords), dtype=np.float32)
    dc = np.asarray(_tensor_to_numpy(sh0), dtype=np.float32)
    if source_positions.ndim != 2 or source_positions.shape[1] != 3:
        raise ValueError(
            f"GLB positions must have shape (N, 3), got {source_positions.shape}."
        )
    if dc.shape != source_positions.shape:
        raise ValueError(
            "GLB DC color coefficients must match positions, got "
            f"{dc.shape} and {source_positions.shape}."
        )
    if source_positions.shape[0] == 0:
        raise ValueError("Cannot write a GLB point preview with zero points.")
    if not bool(np.isfinite(source_positions).all()) or not bool(np.isfinite(dc).all()):
        raise RuntimeError("Cannot write GLB preview from NaN or infinite values.")

    positions = np.empty_like(source_positions, dtype=np.float32)
    positions[:, 0] = source_positions[:, 0]
    positions[:, 1] = source_positions[:, 2]
    positions[:, 2] = -source_positions[:, 1]
    positions = np.ascontiguousarray(positions.astype("<f4", copy=False))
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    colors = np.ascontiguousarray(np.rint(rgb * 255.0).astype(np.uint8))

    position_bytes = positions.tobytes(order="C")
    color_offset = len(position_bytes)
    color_bytes = colors.tobytes(order="C")
    binary_blob = _align4(position_bytes + color_bytes)

    gltf = {
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "max": [float(value) for value in positions.max(axis=0)],
                "min": [float(value) for value in positions.min(axis=0)],
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5121,
                "count": int(colors.shape[0]),
                "normalized": True,
                "type": "VEC3",
            },
        ],
        "asset": {
            "generator": "Modly GaussianGPT point-cloud compatibility preview",
            "version": "2.0",
            "extras": {
                "coordinate_transform": "GaussianGPT (x,y,z) to glTF (x,z,-y)"
            },
        },
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteLength": len(position_bytes),
                "byteOffset": 0,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteLength": len(color_bytes),
                "byteOffset": color_offset,
                "target": 34962,
            },
        ],
        "meshes": [
            {
                "name": "Gaussian centers compatibility preview",
                "primitives": [
                    {
                        "attributes": {"COLOR_0": 1, "POSITION": 0},
                        "mode": 0,
                    }
                ],
            }
        ],
        "nodes": [{"mesh": 0, "name": "GaussianGPT point preview"}],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
    }

    json_chunk = _align4(
        json.dumps(
            gltf,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8"),
        b" ",
    )
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_blob)
    payload = (
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary_blob), b"BIN\x00")
        + binary_blob
    )
    path.write_bytes(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_setup_status() -> dict[str, Any] | None:
    try:
        value = json.loads(SETUP_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _validate_rfc3339_utc(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        r"(?:\.\d+)?(?:Z|\+00:00)",
        value,
    ):
        raise AssetVerificationError(
            f"{field_name} must be an RFC3339 UTC timestamp ending in Z or +00:00."
        )
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise AssetVerificationError(
            f"{field_name} is not a valid RFC3339 UTC timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AssetVerificationError(f"{field_name} must be UTC.")
    return value


def _validate_setup_status() -> dict[str, Any]:
    if SETUP_STATUS_PATH.is_symlink() or not SETUP_STATUS_PATH.is_file():
        raise SetupReadinessError(
            f"Dependency setup evidence is missing at {SETUP_STATUS_PATH}. "
            "Run extension setup before loading GaussianGPT."
        )
    status = _read_setup_status()
    if status is None:
        raise SetupReadinessError("Dependency setup status is unreadable or invalid JSON.")
    if status.get("schema") != "modly.setup-status.v1":
        raise SetupReadinessError(
            "Dependency setup status schema must be 'modly.setup-status.v1'."
        )
    if status.get("extension_id") != EXTENSION_ID:
        raise SetupReadinessError(
            f"Dependency setup status belongs to {status.get('extension_id')!r}, "
            f"not {EXTENSION_ID!r}."
        )
    if (
        status.get("status") != "ready"
        or status.get("code") != "ready"
        or status.get("ready") is not True
    ):
        raise SetupReadinessError(
            "Dependency setup is not ready: "
            f"status={status.get('status')!r}, code={status.get('code')!r}, "
            f"message={status.get('message')!r}."
        )
    if status.get("upstream_commit") != UPSTREAM_COMMIT:
        raise SetupReadinessError(
            "Dependency setup evidence was produced for a different upstream commit."
        )
    lane = status.get("lane")
    toolkit = status.get("actual_cuda_toolkit")
    if not isinstance(lane, dict) or not isinstance(lane.get("lane_id"), str):
        raise SetupReadinessError("Dependency setup status lacks a selected runtime lane.")
    if (
        not isinstance(toolkit, dict)
        or toolkit.get("release") != lane.get("expected_torch_cuda")
    ):
        raise SetupReadinessError(
            "Dependency setup status toolkit does not match the selected runtime lane."
        )
    probe = status.get("probe")
    if not isinstance(probe, dict) or probe.get("ok") is not True:
        raise SetupReadinessError(
            "Dependency setup status lacks successful authoritative native/CUDA probe evidence."
        )
    probe_torch = probe.get("torch")
    if (
        not isinstance(probe_torch, dict)
        or probe_torch.get("cuda_version") != lane.get("expected_torch_cuda")
    ):
        raise SetupReadinessError(
            "Dependency setup probe CUDA version does not match the selected runtime lane."
        )
    pip_check = probe.get("pip_check")
    if not isinstance(pip_check, dict) or pip_check.get("ok") is not True:
        raise SetupReadinessError(
            "Dependency setup status lacks a successful pip-check result."
        )
    recorded_python = status.get("venv_python")
    if not isinstance(recorded_python, str) or not recorded_python.strip():
        raise SetupReadinessError("Dependency setup status has no venv_python.")
    try:
        recorded_resolved = Path(recorded_python).expanduser().resolve(strict=True)
        current_resolved = Path(sys.executable).expanduser().resolve(strict=True)
    except OSError as exc:
        raise SetupReadinessError(
            f"Cannot resolve setup/runtime Python identity: {exc}"
        ) from exc
    if recorded_resolved != current_resolved:
        raise SetupReadinessError(
            f"Dependency setup was validated with {recorded_resolved}, but the "
            f"current runtime is {current_resolved}."
        )
    return status


class GaussianGPTGenerator(BaseGenerator):
    MODEL_ID = MODEL_ID
    DISPLAY_NAME = DISPLAY_NAME
    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return _schema_for_node(DEFAULT_NODE_ID)

    @classmethod
    def capability_params_schema(cls, node_id: str) -> list[dict[str, Any]]:
        if node_id not in NODE_CONFIGS:
            raise ValueError(f"Unknown GaussianGPT node {node_id!r}.")
        return _schema_for_node(node_id)

    def __init__(
        self,
        model_dir: Path | str | None = None,
        outputs_dir: Path | str | None = None,
    ) -> None:
        if model_dir is None:
            raise ValueError(
                "GaussianGPT requires the node-owned model_dir injected by Modly; "
                "environment and extension-local model fallbacks are intentionally disabled."
            )
        normalized_model_dir = Path(model_dir).expanduser().resolve()
        normalized_outputs_dir = Path(outputs_dir or (EXTENSION_DIR / "outputs")).expanduser().resolve()
        super().__init__(normalized_model_dir, normalized_outputs_dir)
        self.model_dir = normalized_model_dir
        self.outputs_dir = normalized_outputs_dir
        self.download_check = DOWNLOAD_CHECK
        self.node_id = getattr(self, "node_id", "")
        self._model: Any | None = None
        self._torch: Any | None = None
        self._loaded_node_id: str | None = None
        self._loaded_model_dir: Path | None = None
        self._verification_cache: dict[
            tuple[str, str],
            tuple[tuple[int, int, int, int], ...],
        ] = {}

    def _node_config(self) -> NodeConfig:
        runtime_node = str(getattr(self, "node_id", "") or "").strip()
        path_node = self.model_dir.name

        if runtime_node and runtime_node not in NODE_CONFIGS:
            expected = ", ".join(sorted(NODE_CONFIGS))
            raise ValueError(
                f"Unknown GaussianGPT runtime node {runtime_node!r}; expected one of: {expected}."
            )
        node_id = runtime_node or path_node
        config = NODE_CONFIGS.get(node_id)
        if config is None:
            expected = ", ".join(sorted(NODE_CONFIGS))
            raise ValueError(
                f"Cannot derive GaussianGPT node from injected model_dir {self.model_dir}; "
                f"its final component must be one of: {expected}."
            )
        if path_node != config.node_id or self.model_dir.parent.name != EXTENSION_ID:
            raise ValueError(
                "GaussianGPT model ownership mismatch: injected model_dir must be exactly "
                f"<models>/{EXTENSION_ID}/{config.node_id}, got {self.model_dir}."
            )
        if runtime_node and runtime_node != path_node:
            raise ValueError(
                f"Runtime node {runtime_node!r} does not own injected model_dir {self.model_dir}."
            )

        self.MODEL_ID = config.model_id
        self.DISPLAY_NAME = config.display_name
        self.download_check = DOWNLOAD_CHECK
        return config

    def _manifest_plan(self, config: NodeConfig) -> list[dict[str, Any]]:
        node = _manifest_node(config.node_id)
        if node.get("weight_owner_id") != config.node_id:
            raise AssetVerificationError(
                f"Manifest weight owner for {config.node_id} must be {config.node_id!r}."
            )
        if node.get("download_check") != DOWNLOAD_CHECK:
            raise AssetVerificationError(
                f"Manifest download_check for {config.node_id} must be {DOWNLOAD_CHECK!r}."
            )
        plan = node.get("https_downloads")
        if not isinstance(plan, list):
            raise AssetVerificationError(
                f"Manifest node {config.node_id} has no operational https_downloads array."
            )
        expected = [dict(item) for item in config.https_downloads]
        if plan != expected:
            raise AssetVerificationError(
                f"Manifest HTTPS plan for {config.node_id} differs from the pinned official asset plan."
            )
        return [dict(item) for item in plan]

    def _verification_snapshot(
        self,
        marker_path: Path,
        plan: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[int, int, int, int], ...]:
        return (
            _file_signature(marker_path),
            *(_file_signature(self.model_dir / item["filename"]) for item in plan),
        )

    def _verify_assets(
        self,
        config: NodeConfig,
        *,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> dict[str, Any]:
        plan = self._manifest_plan(config)
        marker_path = self.model_dir / DOWNLOAD_CHECK
        if marker_path.is_symlink() or not marker_path.is_file():
            raise AssetVerificationError(
                f"{config.model_id} is not ready: Modly readiness marker is missing at "
                f"{marker_path}. Install the complete checkpoint pair from the Models UI."
            )

        for item in plan:
            asset_path = self.model_dir / item["filename"]
            if asset_path.is_symlink() or not asset_path.is_file():
                raise AssetVerificationError(
                    f"{config.model_id} is missing required asset {item['filename']} at "
                    f"{asset_path}. Use the Modly Models UI; setup.py never downloads weights."
                )

        cache_key = (config.node_id, str(self.model_dir))
        snapshot_before = self._verification_snapshot(marker_path, plan)
        if self._verification_cache.get(cache_key) == snapshot_before:
            return {
                "model_id": config.model_id,
                "model_dir": str(self.model_dir),
                "marker_path": str(marker_path),
                "plan_sha256": _plan_sha256(plan),
                "cached": True,
            }

        marker = _read_json_object(marker_path)
        expected_plan_sha = _plan_sha256(plan)
        expected_assets = _expected_marker_assets(plan)
        if type(marker.get("schema_version")) is not int or marker["schema_version"] != MARKER_SCHEMA_VERSION:
            raise AssetVerificationError(
                f"Readiness marker schema_version must be {MARKER_SCHEMA_VERSION}."
            )
        if marker.get("kind") != MARKER_KIND:
            raise AssetVerificationError(f"Readiness marker kind must be {MARKER_KIND!r}.")
        if marker.get("model_id") != config.model_id:
            raise AssetVerificationError(
                f"Readiness marker belongs to {marker.get('model_id')!r}, not {config.model_id!r}."
            )
        if marker.get("plan_sha256") != expected_plan_sha:
            raise AssetVerificationError(
                "Readiness marker plan_sha256 does not match the canonical manifest HTTPS plan."
            )
        if marker.get("assets") != expected_assets:
            raise AssetVerificationError(
                "Readiness marker asset inventory does not exactly match the ordered manifest plan."
            )
        _validate_rfc3339_utc(
            marker.get("verified_at"),
            "Readiness marker verified_at",
        )

        _progress(progress_cb, 4, f"Verifying {config.pair_name} VQ-VAE checkpoint")
        for index, item in enumerate(plan):
            _raise_if_cancelled(cancel_event)
            asset_path = self.model_dir / item["filename"]
            size = asset_path.stat().st_size
            if size != item["size_bytes"]:
                raise AssetVerificationError(
                    f"{item['filename']} has size {size}, expected {item['size_bytes']}; "
                    "the checkpoint is incomplete or corrupt."
                )
            digest = _sha256_file(asset_path, cancel_event)
            if digest != item["sha256"]:
                raise AssetVerificationError(
                    f"{item['filename']} SHA256 is {digest}, expected {item['sha256']}; "
                    "the checkpoint is corrupt."
                )
            _progress(
                progress_cb,
                7 + index * 4,
                f"Verified {item['filename']}",
            )

        snapshot_after = self._verification_snapshot(marker_path, plan)
        if snapshot_after != snapshot_before:
            raise AssetVerificationError(
                "Checkpoint files or readiness marker changed during verification; retry after downloads stop."
            )
        self._verification_cache[cache_key] = snapshot_after
        return {
            "model_id": config.model_id,
            "model_dir": str(self.model_dir),
            "marker_path": str(marker_path),
            "plan_sha256": expected_plan_sha,
            "cached": False,
        }

    def is_downloaded(self) -> bool:
        try:
            self._verify_assets(self._node_config())
        except (AssetVerificationError, OSError, ValueError, RuntimeError):
            return False
        return True

    def readiness_status(self) -> dict[str, Any]:
        try:
            config = self._node_config()
        except (ValueError, RuntimeError) as exc:
            return {
                "ok": False,
                "machine_code": "invalid_node_context",
                "label_hint": "Invalid model directory",
                "reason": str(exc),
            }

        try:
            setup_status = _validate_setup_status()
        except SetupReadinessError as exc:
            return {
                "ok": False,
                "machine_code": "setup_not_ready",
                "label_hint": "Run or repair setup",
                "reason": str(exc),
                "details": {"setup_status_path": str(SETUP_STATUS_PATH)},
            }

        dependency_diagnostics = _dependency_status()
        details: dict[str, Any] = {
            "model_id": config.model_id,
            "model_dir": str(self.model_dir),
            "download_check": DOWNLOAD_CHECK,
            "dependency_imports": dependency_diagnostics,
            "vendor_files": _vendor_status(),
            "upstream_commit": UPSTREAM_COMMIT,
            "setup_status": setup_status,
        }
        missing_vendor = [path for path, present in details["vendor_files"].items() if not present]
        if missing_vendor:
            return {
                "ok": False,
                "machine_code": "missing_vendored_source",
                "label_hint": "Reinstall extension",
                "reason": "Pinned GaussianGPT source is incomplete: " + ", ".join(missing_vendor),
                "details": details,
            }

        try:
            details["asset_verification"] = self._verify_assets(config)
        except (AssetVerificationError, OSError) as exc:
            return {
                "ok": False,
                "machine_code": "assets_not_ready",
                "label_hint": "Install or repair checkpoint pair",
                "reason": str(exc),
                "details": details,
            }

        missing_dependencies = [
            package
            for package, diagnostic in dependency_diagnostics.items()
            if diagnostic.get("ok") is not True
        ]
        if missing_dependencies:
            return {
                "ok": False,
                "machine_code": "missing_dependencies",
                "label_hint": "Runtime lane unavailable",
                "reason": (
                    "GaussianGPT runtime import probe failed: "
                    + "; ".join(
                        f"{package}: "
                        f"{dependency_diagnostics[package].get('error', 'import failed')}"
                        for package in missing_dependencies
                    )
                ),
                "details": details,
            }
        return {
            "ok": True,
            "machine_code": "ready",
            "label_hint": "Ready",
            "reason": f"{config.model_id} source, dependencies, marker, and checkpoint hashes verified.",
            "details": details,
        }

    def _activate_vendor(self) -> None:
        missing = [str(path) for path, present in _vendor_status().items() if not present]
        if missing:
            raise RuntimeError(
                "Pinned GaussianGPT vendor tree is incomplete: " + ", ".join(missing)
            )

        vendor_resolved = VENDOR_ROOT.resolve()
        for root_name in ("model", "utils", "serialization", "conf", "data"):
            loaded = sys.modules.get(root_name)
            origin = getattr(loaded, "__file__", None) if loaded is not None else None
            if origin:
                try:
                    Path(origin).resolve().relative_to(vendor_resolved)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Python module namespace {root_name!r} is already owned by {origin}; "
                        "GaussianGPT must run in its extension venv subprocess."
                    ) from exc

        vendor_text = str(vendor_resolved)
        if vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)

    def _assert_vendored_module(self, module: Any, name: str) -> None:
        origin = getattr(module, "__file__", None)
        if not origin:
            raise RuntimeError(f"Imported {name} has no source origin.")
        try:
            Path(origin).resolve().relative_to(VENDOR_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing non-vendored {name} import from {origin}; expected {VENDOR_ROOT}."
            ) from exc

    def _load_upstream_pair(self, config: NodeConfig) -> tuple[Any, Any]:
        self._activate_vendor()
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            import model.gaussian_gpt as gaussian_gpt_module
            import model.gaussian_vqvae as gaussian_vqvae_module

        self._assert_vendored_module(gaussian_gpt_module, "model.gaussian_gpt")
        self._assert_vendored_module(gaussian_vqvae_module, "model.gaussian_vqvae")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            _log(
                f"{config.model_id}: CUDA is unavailable; CPU loading is allowed but "
                "sampling may be prohibitively slow or unsupported by native dependencies."
            )

        vqvae_path = self.model_dir / config.vqvae_filename
        gpt_path = self.model_dir / config.gpt_filename
        with contextlib.redirect_stdout(sys.stderr):
            vqvae = gaussian_vqvae_module.GaussianVQVAE.load_from_checkpoint(
                str(vqvae_path),
                map_location="cpu",
            ).eval()
            gpt = gaussian_gpt_module.GaussianGPT.load_from_checkpoint(
                str(gpt_path),
                map_location="cpu",
                vqvae=vqvae,
            ).eval()
            gpt = gpt.to(device)
        self._torch = torch
        return gpt, device

    def load(
        self,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> None:
        config = self._node_config()
        if (
            self._model is not None
            and self._loaded_node_id == config.node_id
            and self._loaded_model_dir == self.model_dir
        ):
            return

        _raise_if_cancelled(cancel_event)
        _validate_setup_status()
        self._verify_assets(
            config,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        dependency_diagnostics = _dependency_status()
        missing_dependencies = [
            package
            for package, diagnostic in dependency_diagnostics.items()
            if diagnostic.get("ok") is not True
        ]
        if missing_dependencies:
            raise RuntimeError(
                f"{config.model_id} cannot load; bounded runtime import failures: "
                + "; ".join(
                    f"{package}: "
                    f"{dependency_diagnostics[package].get('error', 'import failed')}"
                    for package in missing_dependencies
                )
            )

        self.unload()
        _progress(
            progress_cb,
            15,
            "Loading GaussianGPT checkpoints (non-interruptible until deserialization returns)",
        )
        _log(
            f"{config.model_id}: loading the verified VQ-VAE/GPT pair. "
            "Checkpoint deserialization and device transfer cannot be interrupted safely."
        )
        _raise_if_cancelled(cancel_event)
        model, device = self._load_upstream_pair(config)
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            del model
            gc.collect()
            raise GenerationCancelled()

        self._model = model
        self._loaded_node_id = config.node_id
        self._loaded_model_dir = self.model_dir
        _progress(progress_cb, 40, f"Loaded {config.model_id} on {device}")
        _log(f"{config.model_id}: verified checkpoints loaded on {device}.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        model = self._model
        torch_module = self._torch
        self._model = None
        self._torch = None
        self._loaded_node_id = None
        self._loaded_model_dir = None
        if model is not None:
            del model
        gc.collect()
        if torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
            except Exception:
                pass

    def _sample_scene(self, sampling: Mapping[str, Any]) -> Any:
        if self._model is None:
            raise RuntimeError("GaussianGPT model is not loaded.")
        with contextlib.redirect_stdout(sys.stderr):
            vqvae = getattr(self._model, "vqvae", None)
            if vqvae is not None and hasattr(vqvae, "set_background_color"):
                vqvae.set_background_color(sampling["background_color"])
            scenes = self._model.sample(
                max_length=None,
                num_samples=1,
                temperature=sampling["temperature"],
                top_k=None if sampling["top_k"] == 0 else sampling["top_k"],
                top_p=sampling["top_p"],
                condition=None,
                return_lengths=False,
                seed=sampling["seed"],
            )
        if isinstance(scenes, tuple):
            scenes = scenes[0]
        if not isinstance(scenes, (list, tuple)) or len(scenes) != 1:
            raise RuntimeError(
                "GaussianGPT.sample() must return exactly one scene for fixed num_samples=1."
            )
        return scenes[0]

    def _write_pt_payload(self, payload: Mapping[str, Any], path: Path) -> None:
        if self._torch is None:
            import torch
        else:
            torch = self._torch
        with contextlib.redirect_stdout(sys.stderr):
            torch.save(dict(payload), path)

    def _write_inria_ply(self, payload: Mapping[str, Any], path: Path) -> int:
        self._activate_vendor()
        with contextlib.redirect_stdout(sys.stderr):
            import data.photoshape as photoshape_module
        self._assert_vendored_module(photoshape_module, "data.photoshape")
        with contextlib.redirect_stdout(sys.stderr):
            return int(
                photoshape_module.save_inria_ply(
                    output_path=str(path),
                    coords=payload["coords"],
                    sh0=payload["sh0"],
                    opacities=payload["opacities"],
                    scales=payload["scales"],
                    quats=payload["quats"],
                    sh=payload.get("sh"),
                )
            )

    def _render_orbit(
        self,
        scene: Any,
        path: Path,
        background_color: str,
    ) -> None:
        if not hasattr(scene, "render_and_save_trajectory"):
            raise RuntimeError("Gaussian scene does not support upstream orbit rendering.")
        self._activate_vendor()
        with contextlib.redirect_stdout(sys.stderr):
            import utils.render as render_module
        self._assert_vendored_module(render_module, "utils.render")
        with contextlib.redirect_stdout(sys.stderr):
            preview_scene = render_module.center_scene_aabb(scene)
            preview_scene.render_and_save_trajectory(
                str(path),
                background_color=background_color,
            )

    def _allocate_run_paths(self, config: NodeConfig) -> tuple[Path, Path]:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for _attempt in range(100):
            token = uuid.uuid4().hex
            run_name = f"gaussiangpt_{config.pair_name}_{stamp}_{token[:8]}"
            final_dir = self.outputs_dir / run_name
            stage_dir = self.outputs_dir / f".{run_name}.{token}.tmp"
            try:
                stage_dir.mkdir(mode=0o700)
            except FileExistsError:
                continue
            if final_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
                continue
            return stage_dir, final_dir
        raise RuntimeError("Could not allocate a unique GaussianGPT output run directory.")

    def generate(
        self,
        image_bytes: bytes,
        params: Mapping[str, Any] | None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> Path:
        config = self._node_config()
        if image_bytes:
            raise ValueError(
                f"{config.model_id} is unconditional and does not accept image input."
            )
        sampling = _validate_params(params)

        _log(
            f"{config.model_id}: starting unconditional generation with "
            f"temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
            f"top_k={sampling['top_k']}, seed={sampling['seed']}, num_samples=1."
        )
        _progress(progress_cb, 1, "Validating GaussianGPT readiness")
        _raise_if_cancelled(cancel_event)
        self.load(progress_cb=progress_cb, cancel_event=cancel_event)

        _progress(
            progress_cb,
            45,
            "Sampling Gaussian scene (non-interruptible until GaussianGPT.sample returns)",
        )
        _log(
            f"{config.model_id}: entering GaussianGPT.sample(); upstream autoregressive "
            "sampling has no safe mid-sequence cancellation hook."
        )
        _raise_if_cancelled(cancel_event)
        scene = self._sample_scene(sampling)
        _raise_if_cancelled(cancel_event)

        payload = _canonical_scene_payload(scene)
        gaussian_count = _validate_scene_payload(payload)
        verification = self._verify_assets(config)
        stage_dir, final_dir = self._allocate_run_paths(config)

        pt_path = stage_dir / "scene.pt"
        ply_path = stage_dir / "scene.ply"
        glb_path = stage_dir / "preview.glb"
        orbit_path = stage_dir / "orbit.gif"
        metadata_path = stage_dir / "metadata.json"

        try:
            _progress(progress_cb, 74, "Writing canonical Gaussian PT payload")
            _raise_if_cancelled(cancel_event)
            self._write_pt_payload(payload, pt_path)

            _progress(
                progress_cb,
                80,
                "Writing official INRIA Gaussian PLY (non-interruptible until exporter returns)",
            )
            _raise_if_cancelled(cancel_event)
            ply_count = self._write_inria_ply(payload, ply_path)
            _raise_if_cancelled(cancel_event)
            if ply_count != gaussian_count:
                raise RuntimeError(
                    f"INRIA PLY exporter wrote {ply_count} Gaussians, expected {gaussian_count}."
                )

            _progress(progress_cb, 86, "Writing GLB point-cloud compatibility preview")
            _raise_if_cancelled(cancel_event)
            _write_point_cloud_glb(payload["coords"], payload["sh0"], glb_path)

            orbit_filename: str | None = None
            if sampling["render_preview"]:
                _progress(
                    progress_cb,
                    90,
                    "Rendering orbit GIF (non-interruptible until upstream renderer returns)",
                )
                _log(
                    f"{config.model_id}: entering upstream Gaussian orbit renderer; "
                    "cancellation is checked immediately after it returns."
                )
                _raise_if_cancelled(cancel_event)
                self._render_orbit(scene, orbit_path, sampling["background_color"])
                _raise_if_cancelled(cancel_event)
                if not orbit_path.is_file():
                    raise RuntimeError("Upstream orbit renderer returned without writing orbit.gif.")
                orbit_filename = orbit_path.name

            plan = self._manifest_plan(config)
            metadata = {
                "schema_version": 1,
                "extension_id": EXTENSION_ID,
                "node_id": config.node_id,
                "model_id": config.model_id,
                "upstream": {
                    "repository": "https://github.com/nicolasvonluetzow/GaussianGPT",
                    "commit": UPSTREAM_COMMIT,
                    "license": "MIT",
                },
                "checkpoint_pair": {
                    "name": config.pair_name,
                    "plan_sha256": verification["plan_sha256"],
                    "assets": _expected_marker_assets(plan),
                    "weights_license": "Unknown; not declared by the official checkpoint host",
                },
                "sampling": {
                    "temperature": sampling["temperature"],
                    "top_p": sampling["top_p"],
                    "top_k": sampling["top_k"],
                    "top_k_effective": None if sampling["top_k"] == 0 else sampling["top_k"],
                    "seed": sampling["seed"],
                    "num_samples": 1,
                    "background_color": sampling["background_color"],
                    "render_preview": sampling["render_preview"],
                },
                "gaussian_count": gaussian_count,
                "created_at": _utc_now(),
                "artifacts": {
                    "canonical_pt": {
                        "filename": pt_path.name,
                        "representation": "Canonical GaussianGPT Gaussian tensor payload",
                        "fields": sorted(payload),
                    },
                    "inria_ply": {
                        "filename": ply_path.name,
                        "representation": "INRIA-style Gaussian splat payload",
                    },
                    "glb_preview": {
                        "filename": glb_path.name,
                        "representation": (
                            "POINTS compatibility preview of Gaussian centers and "
                            "DC color only, transformed (x,y,z) to (x,z,-y)"
                        ),
                        "fidelity_warning": (
                            "This GLB is neither a Gaussian-splat representation nor a surface mesh. "
                            "It omits opacity, covariance, scale, rotation, and higher-order SH fidelity."
                        ),
                    },
                    "orbit_preview": (
                        {
                            "filename": orbit_filename,
                            "representation": "Upstream Gaussian renderer orbit GIF",
                        }
                        if orbit_filename
                        else None
                    ),
                    "metadata": {"filename": metadata_path.name},
                },
                "cancellation_boundaries": {
                    "interruptible": [
                        "checkpoint hashing between read chunks",
                        "before and after checkpoint loading",
                        "before and after sampling",
                        "between artifact writes",
                        "before and after orbit rendering",
                        "before atomic publication",
                    ],
                    "non_interruptible": [
                        "Lightning checkpoint deserialization and device transfer",
                        "GaussianGPT.sample() autoregressive sequence",
                        "official INRIA PLY export",
                        "upstream orbit rendering",
                    ],
                },
            }
            _progress(progress_cb, 96, "Writing GaussianGPT metadata")
            _raise_if_cancelled(cancel_event)
            _write_json(metadata_path, metadata)

            expected_files = {pt_path, ply_path, glb_path, metadata_path}
            if sampling["render_preview"]:
                expected_files.add(orbit_path)
            missing_outputs = [path.name for path in expected_files if not path.is_file()]
            if missing_outputs:
                raise RuntimeError(
                    "GaussianGPT artifact bundle is incomplete: " + ", ".join(sorted(missing_outputs))
                )

            _raise_if_cancelled(cancel_event)
            os.replace(stage_dir, final_dir)
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise

        result = final_dir / glb_path.name
        _progress(progress_cb, 100, "GaussianGPT artifact bundle complete")
        _log(
            f"{config.model_id}: published atomic bundle at {final_dir}. Returning {result}; "
            "the GLB is a point compatibility preview, not Gaussian or mesh fidelity."
        )
        return result

    def _auto_download(self) -> None:
        raise RuntimeError(
            "GaussianGPT weights are manifest-owned HTTPS assets. Install the complete "
            "node checkpoint pair from the Modly Models UI; generator.py never downloads files."
        )
