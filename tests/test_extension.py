from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load_local_module("gaussiangpt_generator_test", ROOT / "generator.py")
setup_module = load_local_module("gaussiangpt_setup_test", ROOT / "setup.py")


class ManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

    def test_identity_nodes_and_unconditional_contract(self):
        manifest = self.manifest
        self.assertEqual(manifest["id"], "gaussiangpt")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(manifest["author"], "DrHepa")
        self.assertEqual(
            manifest["source"],
            "https://github.com/DrHepa/modly-gaussiangpt-extension",
        )
        self.assertEqual(manifest["generator_class"], "GaussianGPTGenerator")
        self.assertNotIn("vram_gb", manifest)
        self.assertNotIn("recommended_vram_gb", manifest)
        self.assertEqual(manifest["outputs"][0]["formats"], ["glb"])
        self.assertEqual(
            {item["format"] for item in manifest["outputs"][0]["sidecars"]},
            {"pt", "ply", "gif", "json"},
        )
        nodes = {node["id"]: node for node in manifest["nodes"]}
        self.assertEqual(set(nodes), {"generate-vfront", "generate-both"})
        for node_id, node in nodes.items():
            with self.subTest(node_id=node_id):
                self.assertTrue(node["permanent"])
                self.assertEqual(node["input"], "none")
                self.assertEqual(node["output"], "mesh")
                self.assertEqual(node["output_formats"], ["glb"])
                self.assertEqual(
                    set(node["sidecar_formats"]),
                    {"pt", "ply", "gif", "json"},
                )
                self.assertEqual(node["weight_owner_id"], node_id)
                self.assertEqual(node["download_check"], ".modly/https-assets-ready.json")
                self.assertEqual(node["num_samples"], 1)
                self.assertEqual(node["fixed_params"], {"num_samples": 1})

    def test_exact_operational_https_asset_plans(self):
        nodes = {node["id"]: node for node in self.manifest["nodes"]}
        expected = {
            "generate-vfront": [
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
            ],
            "generate-both": [
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
            ],
        }
        for node_id, plan in expected.items():
            self.assertEqual(nodes[node_id]["https_downloads"], plan)
            for item in plan:
                self.assertEqual(
                    set(item),
                    {"url", "filename", "size_bytes", "sha256"},
                )

    def test_params_exclude_conditioning_and_sample_count(self):
        for node in self.manifest["nodes"]:
            schema = {item["id"]: item for item in node["params_schema"]}
            self.assertEqual(
                set(schema),
                {
                    "temperature",
                    "top_p",
                    "top_k",
                    "seed",
                    "background_color",
                    "render_preview",
                },
            )
            self.assertEqual(schema["temperature"]["default"], 0.9)
            self.assertEqual(schema["temperature"]["min"], 0.0)
            self.assertEqual(schema["top_p"]["default"], 0.9)
            self.assertEqual(schema["top_k"]["default"], 0)
            self.assertNotIn("max", schema["top_k"])
            self.assertEqual(schema["seed"]["default"], 0)
            self.assertEqual(schema["seed"]["max"], 2147483646)


class MarkerAndParameterTests(unittest.TestCase):
    def canonical_model_dir(self, root: Path) -> Path:
        path = root / "models" / "gaussiangpt" / "generate-vfront"
        path.mkdir(parents=True)
        return path

    def test_plan_hash_uses_exact_canonical_json_for_both_pairs(self):
        expected_hash_by_node = {
            "generate-vfront": "7e2c3c305c5eef0d6f75558486299a71934909e95af580fed0ebc3dc816df994",
            "generate-both": "5e7b81f5d8a30772a29f1bd48197a05c6842fce7d1469fcfe088a938897ba2b1",
        }
        for node_id, expected_hash in expected_hash_by_node.items():
            with self.subTest(node_id=node_id):
                plan = [
                    dict(item)
                    for item in generator.NODE_CONFIGS[node_id].https_downloads
                ]
                canonical = json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                self.assertEqual(generator._canonical_plan_json(plan), canonical)
                self.assertEqual(
                    hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    expected_hash,
                )
                self.assertEqual(generator._plan_sha256(plan), expected_hash)
                self.assertNotEqual(
                    generator._plan_sha256(plan),
                    generator._plan_sha256(list(reversed(plan))),
                )

    def write_small_ready_pair(self, model_dir: Path):
        values = {"a.ckpt": b"alpha", "b.ckpt": b"bravo"}
        plan = [
            {
                "url": "https://example.invalid/a.ckpt",
                "filename": "a.ckpt",
                "size_bytes": len(values["a.ckpt"]),
                "sha256": hashlib.sha256(values["a.ckpt"]).hexdigest(),
            },
            {
                "url": "https://example.invalid/b.ckpt",
                "filename": "b.ckpt",
                "size_bytes": len(values["b.ckpt"]),
                "sha256": hashlib.sha256(values["b.ckpt"]).hexdigest(),
            },
        ]
        for filename, value in values.items():
            (model_dir / filename).write_bytes(value)
        marker = {
            "schema_version": 1,
            "kind": "modly.https-assets.ready",
            "model_id": "gaussiangpt/generate-vfront",
            "plan_sha256": generator._plan_sha256(plan),
            "assets": generator._expected_marker_assets(plan),
            "verified_at": "2026-07-11T00:00:00+00:00",
        }
        marker_path = model_dir / ".modly" / "https-assets-ready.json"
        marker_path.parent.mkdir()
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        return plan, values

    def test_missing_marker_and_corrupt_asset_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = self.canonical_model_dir(root)
            gen = generator.GaussianGPTGenerator(model_dir, root / "outputs")
            config = gen._node_config()
            with self.assertRaisesRegex(generator.AssetVerificationError, "marker is missing"):
                gen._verify_assets(config)

            plan, values = self.write_small_ready_pair(model_dir)
            with mock.patch.object(gen, "_manifest_plan", return_value=plan):
                verified = gen._verify_assets(config)
                self.assertFalse(verified["cached"])
                self.assertTrue(gen._verify_assets(config)["cached"])
                gen._verification_cache.clear()
                (model_dir / "b.ckpt").write_bytes(b"BRAVO")
                with self.assertRaisesRegex(generator.AssetVerificationError, "SHA256"):
                    gen._verify_assets(config)

            self.assertEqual(values["a.ckpt"], b"alpha")

    def test_marker_inventory_and_model_id_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = self.canonical_model_dir(root)
            gen = generator.GaussianGPTGenerator(model_dir, root / "outputs")
            config = gen._node_config()
            plan, _values = self.write_small_ready_pair(model_dir)
            marker_path = model_dir / ".modly" / "https-assets-ready.json"
            marker = json.loads(marker_path.read_text())
            marker["model_id"] = "gaussiangpt/generate-both"
            marker_path.write_text(json.dumps(marker))
            with mock.patch.object(gen, "_manifest_plan", return_value=plan):
                with self.assertRaisesRegex(generator.AssetVerificationError, "belongs to"):
                    gen._verify_assets(config)
            marker["model_id"] = config.model_id
            marker["assets"] = list(reversed(marker["assets"]))
            marker_path.write_text(json.dumps(marker))
            with mock.patch.object(gen, "_manifest_plan", return_value=plan):
                with self.assertRaisesRegex(generator.AssetVerificationError, "inventory"):
                    gen._verify_assets(config)

    def test_marker_verified_at_requires_rfc3339_utc(self):
        valid = (
            "2026-07-11T12:34:56Z",
            "2026-07-11T12:34:56.123456+00:00",
        )
        for value in valid:
            with self.subTest(value=value):
                self.assertEqual(
                    generator._validate_rfc3339_utc(value, "verified_at"),
                    value,
                )
        for value in (
            "2026-07-11",
            "2026-07-11T12:34:56",
            "2026-07-11T12:34:56+02:00",
            "not-a-date",
            "",
            None,
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                generator.AssetVerificationError,
                "RFC3339 UTC",
            ):
                generator._validate_rfc3339_utc(value, "verified_at")

    def test_setup_status_must_be_ready_for_current_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path = root / "venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_bytes(b"python")
            status_path = root / "setup-status.json"
            valid = {
                "schema": "modly.setup-status.v1",
                "extension_id": "gaussiangpt",
                "status": "ready",
                "code": "ready",
                "ready": True,
                "upstream_commit": generator.UPSTREAM_COMMIT,
                "venv_python": str(python_path),
                "lane": {
                    "lane_id": "test-cp311-cu130",
                    "expected_torch_cuda": "13.0",
                },
                "actual_cuda_toolkit": {"release": "13.0"},
                "probe": {
                    "ok": True,
                    "torch": {"cuda_version": "13.0"},
                    "pip_check": {"ok": True, "output": "No broken requirements found."},
                },
            }
            status_path.write_text(json.dumps(valid))
            with mock.patch.object(
                generator,
                "SETUP_STATUS_PATH",
                status_path,
            ), mock.patch.object(
                generator.sys,
                "executable",
                str(python_path),
            ):
                self.assertTrue(generator._validate_setup_status()["ready"])
                failed = dict(valid)
                failed["ready"] = False
                failed["status"] = "blocked"
                status_path.write_text(json.dumps(failed))
                with self.assertRaisesRegex(
                    generator.SetupReadinessError,
                    "not ready",
                ):
                    generator._validate_setup_status()

    def test_parameter_defaults_bounds_and_forbidden_controls(self):
        defaults = generator._validate_params({})
        self.assertEqual(defaults["temperature"], 0.9)
        self.assertEqual(defaults["top_p"], 0.9)
        self.assertEqual(defaults["top_k"], 0)
        self.assertEqual(defaults["seed"], 0)
        self.assertEqual(defaults["num_samples"], 1)
        self.assertTrue(defaults["render_preview"])

        greedy = generator._validate_params(
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 100000,
                "seed": 2147483646,
                "background_color": "black",
                "render_preview": "false",
            }
        )
        self.assertEqual(greedy["temperature"], 0.0)
        self.assertFalse(greedy["render_preview"])
        for params in (
            {"temperature": -0.1},
            {"top_p": 0.0},
            {"seed": 2147483647},
            {"top_k": -1},
            {"background_color": "blue"},
            {"num_samples": 2},
            {"prompt": "room"},
            {"image": "x"},
            {"model_variant": "both"},
            {"max_length": 1024},
            {"output_name": "scene"},
            {"output-name": "scene"},
            {"unexpected_control": True},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                generator._validate_params(params)

    def test_modly_host_compatibility_envelope_is_validated_and_ignored(self):
        self.assertEqual(
            generator.HOST_COMPAT_PARAMS,
            {"remesh", "enable_texture", "texture_resolution"},
        )
        self.assertTrue(generator.HOST_COMPAT_PARAMS.isdisjoint(generator.ALLOWED_PARAMS))
        defaults = generator._validate_params({})

        failed_request_envelope = generator._validate_params(
            {
                "remesh": "none",
                "enable_texture": False,
                "texture_resolution": 1024,
            }
        )
        self.assertEqual(failed_request_envelope, defaults)

        backend_default_with_non_neutral_values = generator._validate_params(
            {
                "remesh": "quad",
                "enable_texture": True,
                "texture_resolution": 2048,
            }
        )
        self.assertEqual(backend_default_with_non_neutral_values, defaults)

        self.assertEqual(
            set(backend_default_with_non_neutral_values),
            {
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "background_color",
                "render_preview",
                "num_samples",
            },
        )

    def test_modly_host_compatibility_envelope_rejects_malformed_values(self):
        for params in (
            {"remesh": "voxel"},
            {"remesh": None},
            {"enable_texture": "false"},
            {"enable_texture": 0},
            {"texture_resolution": True},
            {"texture_resolution": 0},
            {"texture_resolution": 1024.0},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                generator._validate_params(params)

    def test_modly_host_compatibility_envelope_preserves_unknown_key_rejection(self):
        with self.assertRaisesRegex(ValueError, "prompt"):
            generator._validate_params(
                {
                    "remesh": "none",
                    "enable_texture": False,
                    "texture_resolution": 1024,
                    "prompt": "room",
                }
            )

    def test_runtime_identity_cannot_cross_weight_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models" / "gaussiangpt"
            vfront = root / "generate-vfront"
            vfront.mkdir(parents=True)
            gen = generator.GaussianGPTGenerator(vfront, Path(temporary) / "out")
            gen.node_id = "generate-both"
            with self.assertRaisesRegex(ValueError, "does not own|ownership mismatch"):
                gen._node_config()


class SerializationAndStdoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest("NumPy is required for serialization unit tests") from exc
        cls.np = np

    def payload(self):
        np = self.np
        return {
            "coords": np.asarray([[0.0, 0.0, 1.0], [1.0, -1.0, 2.0]], dtype=np.float32),
            "sh0": np.asarray([[0.0, 0.1, -0.1], [0.5, -0.5, 0.0]], dtype=np.float32),
            "opacities": np.asarray([[0.0], [1.0]], dtype=np.float32),
            "scales": np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=np.float32),
            "quats": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        }

    def test_glb_writer_is_deterministic_points_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "a.glb"
            second = Path(temporary) / "b.glb"
            payload = self.payload()
            generator._write_point_cloud_glb(payload["coords"], payload["sh0"], first)
            generator._write_point_cloud_glb(payload["coords"], payload["sh0"], second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            raw = first.read_bytes()
            self.assertEqual(raw[:4], b"glTF")
            version, total = struct.unpack("<II", raw[4:12])
            self.assertEqual(version, 2)
            self.assertEqual(total, len(raw))
            json_length, kind = struct.unpack("<I4s", raw[12:20])
            self.assertEqual(kind, b"JSON")
            gltf = json.loads(raw[20 : 20 + json_length].decode("utf-8"))
            self.assertEqual(gltf["meshes"][0]["primitives"][0]["mode"], 0)
            self.assertIn("compatibility preview", gltf["asset"]["generator"])
            self.assertEqual(
                gltf["asset"]["extras"]["coordinate_transform"],
                "GaussianGPT (x,y,z) to glTF (x,z,-y)",
            )
            self.assertEqual(gltf["accessors"][0]["min"], [0.0, 1.0, 0.0])
            self.assertEqual(gltf["accessors"][0]["max"], [1.0, 2.0, 1.0])
            binary_header = 20 + json_length
            binary_length, binary_kind = struct.unpack(
                "<I4s",
                raw[binary_header : binary_header + 8],
            )
            self.assertEqual(binary_kind, b"BIN\x00")
            binary = raw[binary_header + 8 : binary_header + 8 + binary_length]
            self.assertEqual(
                struct.unpack("<6f", binary[:24]),
                (0.0, 1.0, -0.0, 1.0, 2.0, 1.0),
            )
            self.assertNotIn("opacity", json.dumps(gltf).lower())

    def test_nonfinite_payload_and_glb_are_rejected(self):
        payload = self.payload()
        payload["coords"][0, 0] = self.np.nan
        with self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
            generator._validate_scene_payload(payload)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bad.glb"
            with self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
                generator._write_point_cloud_glb(payload["coords"], payload["sh0"], output)
            self.assertFalse(output.exists())

    def test_real_orbit_path_centers_a_derived_scene_only(self):
        import types

        class Scene:
            def __init__(self, coordinates=None):
                self.coordinates = list(coordinates or [10.0, 20.0, 30.0])

            def render_and_save_trajectory(self, path, background_color="white"):
                Path(path).write_bytes(b"GIF89a")

        original = Scene()
        centered_objects = []

        def center_scene_aabb(_scene):
            centered = Scene([0.0, 0.0, 0.0])
            centered_objects.append(centered)
            return centered

        utils_package = types.ModuleType("utils")
        utils_package.__path__ = []
        render_module = types.ModuleType("utils.render")
        render_module.center_scene_aabb = center_scene_aabb
        utils_package.render = render_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models" / "gaussiangpt" / "generate-vfront"
            model_dir.mkdir(parents=True)
            gen = generator.GaussianGPTGenerator(model_dir, root / "out")
            output = root / "orbit.gif"
            with mock.patch.object(
                gen,
                "_activate_vendor",
            ), mock.patch.object(
                gen,
                "_assert_vendored_module",
            ), mock.patch.dict(
                sys.modules,
                {"utils": utils_package, "utils.render": render_module},
            ):
                gen._render_orbit(original, output, "black")
            self.assertEqual(original.coordinates, [10.0, 20.0, 30.0])
            self.assertEqual(len(centered_objects), 1)
            self.assertIsNot(centered_objects[0], original)
            self.assertEqual(centered_objects[0].coordinates, [0.0, 0.0, 0.0])
            self.assertTrue(output.is_file())

    class FakeVQVAE:
        def __init__(self):
            self.background = None

        def set_background_color(self, value):
            self.background = value

    class FakeScene:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return self._payload

        def render_and_save_trajectory(self, path, background_color="white"):
            print("upstream render stdout")
            Path(path).write_bytes(b"GIF89a")

    class FakeModel:
        def __init__(self, scene, event=None, fail=False):
            self.scene = scene
            self.event = event
            self.fail = fail
            self.vqvae = SerializationAndStdoutTests.FakeVQVAE()
            self.calls = []

        def sample(self, **kwargs):
            print("upstream sample stdout")
            self.calls.append(kwargs)
            if self.event is not None:
                self.event.set()
            if self.fail:
                raise RuntimeError("synthetic upstream failure")
            return [self.scene]

    class FakeGenerator(generator.GaussianGPTGenerator):
        def __init__(self, model_dir, outputs_dir, model):
            super().__init__(model_dir, outputs_dir)
            self.fake_model = model

        def load(self, progress_cb=None, cancel_event=None):
            generator._raise_if_cancelled(cancel_event)
            self._model = self.fake_model
            self._loaded_node_id = "generate-vfront"
            self._loaded_model_dir = self.model_dir

        def _verify_assets(self, config, **kwargs):
            plan = [dict(item) for item in config.https_downloads]
            return {
                "model_id": config.model_id,
                "model_dir": str(self.model_dir),
                "marker_path": str(self.model_dir / generator.DOWNLOAD_CHECK),
                "plan_sha256": generator._plan_sha256(plan),
                "cached": True,
            }

        def _write_pt_payload(self, payload, path):
            path.write_bytes(b"canonical-fake-pt")

        def _write_inria_ply(self, payload, path):
            path.write_bytes(b"ply\n")
            return generator._validate_scene_payload(payload)

        def _render_orbit(self, scene, path, background_color):
            with contextlib.redirect_stdout(sys.stderr):
                scene.render_and_save_trajectory(path, background_color=background_color)

    def fake_generator(self, root, *, event=None, fail=False):
        model_dir = root / "models" / "gaussiangpt" / "generate-vfront"
        model_dir.mkdir(parents=True)
        scene = self.FakeScene(self.payload())
        model = self.FakeModel(scene, event=event, fail=fail)
        gen = self.FakeGenerator(model_dir, root / "outputs", model)
        return gen, model

    def test_atomic_bundle_metadata_and_stdout_cleanliness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gen, model = self.fake_generator(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = gen.generate(
                    b"",
                    {
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "top_k": 0,
                        "seed": 7,
                        "background_color": "black",
                        "render_preview": "true",
                    },
                )
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("upstream sample stdout", stderr.getvalue())
            self.assertEqual(result.name, "preview.glb")
            run_dir = result.parent
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"scene.pt", "scene.ply", "preview.glb", "orbit.gif", "metadata.json"},
            )
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["model_id"], "gaussiangpt/generate-vfront")
            self.assertEqual(metadata["gaussian_count"], 2)
            self.assertEqual(metadata["sampling"]["num_samples"], 1)
            self.assertIn("neither a Gaussian-splat", metadata["artifacts"]["glb_preview"]["fidelity_warning"])
            self.assertEqual(model.calls[0]["num_samples"], 1)
            self.assertIsNone(model.calls[0]["top_k"])
            self.assertEqual(model.vqvae.background, "black")
            self.assertFalse(list((root / "outputs").glob(".*.tmp")))

    def test_upstream_error_keeps_stdout_clean_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gen, _model = self.fake_generator(root, fail=True)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(RuntimeError, "synthetic upstream failure"):
                    gen.generate(b"", {"render_preview": "false"})
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("upstream sample stdout", stderr.getvalue())
            self.assertFalse((root / "outputs").exists())

    def test_cancellation_before_and_after_noninterruptible_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = threading.Event()
            gen, _model = self.fake_generator(root)
            event.set()
            with self.assertRaises(generator.GenerationCancelled):
                gen.generate(b"", {"render_preview": "false"}, cancel_event=event)
            self.assertFalse((root / "outputs").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = threading.Event()
            gen, _model = self.fake_generator(root, event=event)
            with self.assertRaises(generator.GenerationCancelled):
                gen.generate(b"", {"render_preview": "false"}, cancel_event=event)
            self.assertFalse((root / "outputs").exists())


class SetupContractTests(unittest.TestCase):
    def make_extension(self, root: Path) -> Path:
        ext = root / "gaussiangpt"
        (ext / "vendor" / "GaussianGPT" / "model").mkdir(parents=True)
        (ext / "setup.py").write_text("# setup fixture\n")
        (ext / "requirements.txt").write_text("lightning==2.5.2\n")
        (ext / "manifest.json").write_text(json.dumps({"id": "gaussiangpt"}))
        (ext / "UPSTREAM.json").write_text(
            json.dumps({"commit": setup_module.UPSTREAM_COMMIT})
        )
        (ext / "vendor" / "GaussianGPT" / "LICENSE").write_text("MIT")
        (ext / "vendor" / "GaussianGPT" / "README.md").write_text("GaussianGPT")
        (ext / "vendor" / "GaussianGPT" / "model" / "gaussian_gpt.py").write_text("")
        (ext / "vendor" / "GaussianGPT" / "model" / "gaussian_vqvae.py").write_text("")
        return ext

    @staticmethod
    def canonical_probe_output(payload):
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    @staticmethod
    def successful_cusparselt_probe():
        return {
            "schema": setup_module.CUSPARSELT_PROBE_SCHEMA,
            "ok": True,
            "runtime": {
                "implementation": "cpython",
                "python_version": [3, 11, 9],
                "system": "linux",
                "machine": "aarch64",
            },
            "package": {
                "name": "nvidia-cusparselt-cu13",
                "version": "0.8.0",
            },
            "torch": {
                "version": setup_module.CUSPARSELT_TORCH_DISTRIBUTION_VERSION,
                "cusparselt_requires_dist": [
                    setup_module.CUSPARSELT_TORCH_REQUIREMENT
                ],
            },
            "wheel": {
                "path": "nvidia_cusparselt_cu13-0.8.0.dist-info/WHEEL",
                "tags": [setup_module.CUSPARSELT_WHEEL_TAG],
            },
            "library": {
                "relative_path": setup_module.CUSPARSELT_LIBRARY_PATH,
                "resolved_path": (
                    "/target/venv/lib/python3.11/site-packages/"
                    + setup_module.CUSPARSELT_LIBRARY_PATH
                ),
                "size": 123,
                "record": {"sha256": "A" * 43, "size": 123},
                "elf": {"class": 2, "data": 1, "machine": 183},
                "load": {
                    "mode": 2,
                    "flags": ["RTLD_NOW", "RTLD_LOCAL"],
                    "symbols": list(setup_module.CUSPARSELT_REQUIRED_SYMBOLS),
                },
                "properties": [
                    {"enum": 0, "status": 0, "value": 0},
                    {"enum": 1, "status": 0, "value": 8},
                    {"enum": 2, "status": 0, "value": 0},
                ],
                "version": "0.8.0",
            },
        }

    def test_json_contract_and_weight_source_flags_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.parse_setup_config(
                [
                    json.dumps(
                        {
                            "python_exe": sys.executable,
                            "ext_dir": str(ext),
                            "gpu_sm": 121,
                            "cuda_version": 128,
                            "validate_only": True,
                            "no_install": True,
                        }
                    )
                ]
            )
            self.assertEqual(config.ext_dir, ext.resolve())
            self.assertTrue(config.validate_only)
            self.assertTrue(config.no_install)
            for payload in (
                {"ext_dir": str(ext), "download_models": False},
                {"ext_dir": str(ext), "download_weights": True},
                {"ext_dir": str(ext), "model_dir": "/tmp/models"},
                {"ext_dir": str(ext), "clone_source": True},
            ):
                with self.subTest(payload=payload), self.assertRaisesRegex(
                    setup_module.SetupContractError, "rejects"
                ):
                    setup_module.parse_setup_config([json.dumps(payload)])
            with self.assertRaises(setup_module.SetupContractError):
                setup_module.parse_setup_config(
                    [json.dumps({"ext_dir": str(ext)}), "--validate-only"]
                )

    def test_validate_only_writes_blocked_report_without_installing(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
                cuda_version=128,
                gpu_sm=121,
                validate_only=True,
                no_install=True,
            )
            with mock.patch.object(
                setup_module, "normalized_platform", return_value=("linux", "aarch64")
            ), mock.patch.object(
                setup_module, "probe_python_version", return_value=(3, 11, 9)
            ), mock.patch.object(
                setup_module,
                "probe_cuda_toolkits",
                return_value=[
                    {
                        "cuda_home": "/usr/local/cuda-12.8",
                        "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
                        "release": "12.8",
                        "driver_cuda_hint": 128,
                    }
                ],
            ), mock.patch.object(
                setup_module,
                "install_dependencies",
                side_effect=AssertionError("install must not run"),
            ):
                self.assertEqual(setup_module.execute(config), 0)
            status = json.loads(config.status_path.read_text())
            self.assertEqual(status["status"], "blocked")
            self.assertFalse(status["ready"])
            self.assertEqual(status["code"], "venv_missing")
            self.assertFalse(status["downloads"]["weights"])
            self.assertFalse(status["downloads"]["gaussiangpt_source"])
            self.assertFalse(status["downloads"]["dependencies_only_during_install"])
            self.assertEqual(status["driver_cuda_hint"], 128)
            self.assertEqual(status["actual_cuda_toolkit"]["release"], "12.8")
            self.assertFalse(config.venv_dir.exists())

    def test_no_install_missing_venv_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
                cuda_version=128,
                no_install=True,
            )
            with mock.patch.object(
                setup_module, "normalized_platform", return_value=("linux", "aarch64")
            ), mock.patch.object(
                setup_module, "probe_python_version", return_value=(3, 11, 9)
            ), mock.patch.object(
                setup_module,
                "probe_cuda_toolkits",
                return_value=[
                    {
                        "cuda_home": "/usr/local/cuda-12.8",
                        "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
                        "release": "12.8",
                        "driver_cuda_hint": 128,
                    }
                ],
            ):
                self.assertEqual(setup_module.execute(config), 1)
            status = json.loads(config.status_path.read_text())
            self.assertEqual(status["status"], "blocked")
            self.assertEqual(status["code"], "venv_missing")
            self.assertFalse(status["ready"])

    def test_install_plan_uses_only_local_minkowski_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
                cuda_version=128,
            )
            lane = setup_module.LANES[("linux", "aarch64", 128)]
            wheel = (
                config.build_deps_dir
                / "minkowski"
                / "fingerprint"
                / "minkowskiengine-0.5.4-cp311-cp311-linux_aarch64.whl"
            )
            wheel.parent.mkdir(parents=True)
            wheel.write_bytes(b"wheel")
            openblas = {
                "cache_key": "openblas-key",
                "cache_dir": str(
                    config.build_deps_dir / "openblas" / "openblas-key"
                ),
                "artifact": {"archive_sha256": "a" * 64},
            }
            minkowski = {
                "wheel": {"path": str(wheel)},
                "cache_key": "minkowski-key",
            }
            commands = []

            def capture(_config, command, **kwargs):
                commands.append([str(item) for item in command])
                return CompletedProcess(command, 0, "")

            with mock.patch.object(
                setup_module,
                "run_command",
                side_effect=capture,
            ), mock.patch.object(
                setup_module,
                "ensure_openblas_dependency",
                return_value=openblas,
            ), mock.patch.object(
                setup_module,
                "probe_minkowski_build_identity",
                return_value={
                    "torch_version": "2.7.1+cu128",
                    "torch_cuda": "12.8",
                },
            ), mock.patch.object(
                setup_module,
                "ensure_minkowski_wheel",
                return_value=minkowski,
            ):
                evidence = setup_module.install_dependencies(
                    config,
                    lane,
                    ext / "venv" / "bin" / "python",
                    {
                        "cuda_home": "/usr/local/cuda-12.8",
                        "release": "12.8",
                    },
                )

            joined = "\n".join(" ".join(command) for command in commands)
            self.assertIn("torch==2.7.1", joined)
            self.assertIn("torchvision==0.22.1", joined)
            install = commands[-1]
            self.assertIn("--force-reinstall", install)
            self.assertIn("--no-deps", install)
            self.assertEqual(install[-1], str(wheel.resolve()))
            self.assertNotIn(setup_module.MINKOWSKI_REPOSITORY, " ".join(install))
            self.assertNotIn("git+", " ".join(install))
            self.assertNotIn("kaldir.vc.cit.tum.de", joined)
            self.assertNotIn("vqvae_", joined)
            self.assertNotIn("gpt_vfront.ckpt", joined)
            self.assertNotIn("nicolasvonluetzow/GaussianGPT", joined)
            self.assertEqual(
                evidence["install"]["source"],
                "extension-local-cached-wheel",
            )
            self.assertTrue(evidence["install"]["no_deps"])


    def test_setup_source_has_no_model_downloader(self):
        source = (ROOT / "setup.py").read_text(encoding="utf-8")
        for forbidden in (
            "kaldir.vc.cit.tum.de",
            "vqvae_vfront.ckpt",
            "gpt_vfront.ckpt",
            "vqvae_both.ckpt",
            "gpt_both.ckpt",
            "snapshot_download",
            "huggingface_hub",
            "requests.get",
            "urllib.request",
            "wget ",
            "curl ",
            "git clone",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn("MINKOWSKI_COMMIT", source)
        self.assertIn("MINKOWSKI_BUILD_RISK", source)
        self.assertIn("runtime-native-cuda-probe", source)

    def test_self_contained_native_dependency_pins_and_prohibitions(self):
        self.assertEqual(
            setup_module.OPENBLAS_REPOSITORY,
            "https://github.com/OpenMathLib/OpenBLAS.git",
        )
        self.assertEqual(
            setup_module.OPENBLAS_COMMIT,
            "6c77e5e314474773a7749357b153caba4ec3817d",
        )
        self.assertEqual(
            setup_module.OPENBLAS_BUILD_ARGUMENTS,
            (
                "TARGET=ARMV8",
                "DYNAMIC_ARCH=1",
                "NUM_THREADS=64",
                "NOFORTRAN=1",
                "NO_SHARED=1",
                "ONLY_CBLAS=1",
            ),
        )
        self.assertEqual(
            setup_module.MINKOWSKI_REPOSITORY,
            "https://github.com/alpsaur/MinkowskiEngine.git",
        )
        self.assertEqual(
            setup_module.MINKOWSKI_COMMIT,
            "1a17f71f3158b9e94e90703961695de627f3df08",
        )
        self.assertEqual(
            setup_module.BUILD_DEPS_RELATIVE_PATH,
            Path(".modly") / "setup" / "build-deps",
        )
        source = (ROOT / "setup.py").read_text(encoding="utf-8").lower()
        for forbidden in (
            "apt-get",
            "apt install",
            "sudo ",
            "scipy-openblas",
            "scipy_openblas",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_minkowski_cuda13_patch_counts_and_cuda12_noop(self):
        def write_checkout(root: Path, call_count: int = 3):
            coordinate = root / "src" / "coordinate_map_functors.cuh"
            prefetch = (
                root
                / "src"
                / "3rdparty"
                / "concurrent_unordered_map.cuh"
            )
            prefetch.parent.mkdir(parents=True)
            coordinate.write_text(
                "before\n"
                + setup_module.MINKOWSKI_COORDINATE_OLD
                + "\nafter\n",
                encoding="utf-8",
            )
            calls = "\n".join(
                f"CUDA_TRY(cudaMemPrefetchAsync(ptr{i}, size{i}, dev_id, stream));"
                for i in range(call_count)
            )
            prefetch.write_text(
                setup_module.MINKOWSKI_PREFETCH_INSERT_ANCHOR
                + "\n"
                + calls
                + "\n",
                encoding="utf-8",
            )
            return coordinate, prefetch

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "minkowski"
            coordinate, prefetch = write_checkout(source)
            evidence = setup_module.apply_minkowski_compatibility_patches(
                source,
                cuda_code=130,
            )
            coordinate_text = coordinate.read_text(encoding="utf-8")
            prefetch_text = prefetch.read_text(encoding="utf-8")
            self.assertTrue(evidence["applied"])
            self.assertEqual(evidence["patchset"], setup_module.MINKOWSKI_PATCHSET)
            self.assertNotIn(
                setup_module.MINKOWSKI_COORDINATE_OLD,
                coordinate_text,
            )
            self.assertEqual(
                coordinate_text.count(setup_module.MINKOWSKI_COORDINATE_NEW),
                1,
            )
            self.assertEqual(
                prefetch_text.count(
                    setup_module.MINKOWSKI_COMPAT_PREFETCH_CALL
                ),
                4,
            )
            self.assertEqual(
                prefetch_text.count(
                    setup_module.MINKOWSKI_LEGACY_PREFETCH_CALL
                ),
                2,
            )
            self.assertIn("#if CUDART_VERSION >= 13000", prefetch_text)
            self.assertIn("cudaMemLocation location{}", prefetch_text)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "minkowski"
            coordinate, prefetch = write_checkout(source)
            coordinate_before = coordinate.read_text(encoding="utf-8")
            prefetch_before = prefetch.read_text(encoding="utf-8")
            evidence = setup_module.apply_minkowski_compatibility_patches(
                source,
                cuda_code=128,
            )
            self.assertFalse(evidence["applied"])
            self.assertEqual(evidence["patchset"], "none")
            self.assertEqual(coordinate.read_text(), coordinate_before)
            self.assertEqual(prefetch.read_text(), prefetch_before)

    def test_minkowski_patch_fails_before_mutating_any_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "minkowski"
            coordinate = source / "src" / "coordinate_map_functors.cuh"
            prefetch = (
                source
                / "src"
                / "3rdparty"
                / "concurrent_unordered_map.cuh"
            )
            prefetch.parent.mkdir(parents=True)
            coordinate.write_text(
                setup_module.MINKOWSKI_COORDINATE_OLD,
                encoding="utf-8",
            )
            prefetch.write_text(
                setup_module.MINKOWSKI_PREFETCH_INSERT_ANCHOR
                + "\n"
                + "cudaMemPrefetchAsync(a, b, c, d);\n"
                + "cudaMemPrefetchAsync(e, f, g, h);\n",
                encoding="utf-8",
            )
            coordinate_before = coordinate.read_bytes()
            prefetch_before = prefetch.read_bytes()
            with self.assertRaises(setup_module.SetupContractError) as raised:
                setup_module.apply_minkowski_compatibility_patches(
                    source,
                    cuda_code=130,
                )
            self.assertEqual(
                raised.exception.code,
                "minkowski_patch_anchor_mismatch",
            )
            self.assertEqual(coordinate.read_bytes(), coordinate_before)
            self.assertEqual(prefetch.read_bytes(), prefetch_before)

    def test_minkowski_environment_preserves_search_paths_and_adds_cccl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            openblas = root / "openblas"
            (openblas / "include").mkdir(parents=True)
            (openblas / "lib").mkdir()
            (openblas / "include" / "cblas.h").write_text("header")
            cuda_home = root / "cuda-13.0"
            cccl = (
                cuda_home
                / "targets"
                / "sbsa-linux"
                / "include"
                / "cccl"
            )
            (cccl / "thrust").mkdir(parents=True)
            (cccl / "thrust" / "host_vector.h").write_text("header")
            lane = setup_module.LANES[("linux", "aarch64", 130)]
            with mock.patch.dict(
                setup_module.os.environ,
                {
                    "CPATH": "/prior/cpath",
                    "LIBRARY_PATH": "/prior/library",
                    "NVCC_FLAGS": "-lineinfo",
                },
                clear=True,
            ):
                env, resolved_cccl = setup_module.minkowski_build_environment(
                    lane,
                    {
                        "cuda_home": str(cuda_home),
                        "release": "13.0",
                    },
                    openblas,
                )
            self.assertEqual(resolved_cccl, cccl.resolve())
            self.assertEqual(
                env["CPATH"],
                setup_module.os.pathsep.join(
                    [
                        str(openblas / "include"),
                        str(cccl.resolve()),
                        "/prior/cpath",
                    ]
                ),
            )
            self.assertEqual(
                env["LIBRARY_PATH"],
                setup_module.os.pathsep.join(
                    [str(openblas / "lib"), "/prior/library"]
                ),
            )
            self.assertIn("-lineinfo", env["NVCC_FLAGS"])
            for flag in setup_module.CUDA13_NVCC_FLAGS:
                self.assertIn(flag, env["NVCC_FLAGS"])

    def test_cuda12_minkowski_environment_does_not_require_cccl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            openblas = root / "openblas"
            (openblas / "include").mkdir(parents=True)
            (openblas / "lib").mkdir()
            (openblas / "include" / "cblas.h").write_text("header")
            lane = setup_module.LANES[("linux", "aarch64", 128)]
            with mock.patch.dict(
                setup_module.os.environ,
                {"CPATH": "/prior", "LIBRARY_PATH": "/prior-lib"},
                clear=True,
            ):
                env, cccl = setup_module.minkowski_build_environment(
                    lane,
                    {
                        "cuda_home": str(root / "cuda-12.8"),
                        "release": "12.8",
                    },
                    openblas,
                )
            self.assertIsNone(cccl)
            self.assertEqual(
                env["CPATH"],
                f"{openblas / 'include'}{setup_module.os.pathsep}/prior",
            )
            self.assertEqual(
                env["LIBRARY_PATH"],
                f"{openblas / 'lib'}{setup_module.os.pathsep}/prior-lib",
            )

    def test_openblas_cache_requires_static_provenance_matched_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
            )
            identity = setup_module.openblas_cache_identity()
            cache_key = setup_module.canonical_sha256(identity)
            cache = config.build_deps_dir / "openblas" / cache_key
            prefix = cache / "prefix"
            (prefix / "include").mkdir(parents=True)
            (prefix / "lib").mkdir()
            (prefix / "include" / "cblas.h").write_text("header")
            archive = (
                prefix
                / "lib"
                / f"libopenblasp-r{setup_module.OPENBLAS_VERSION}.a"
            )
            archive.write_bytes(b"static-archive")
            (prefix / "lib" / "libopenblas.a").symlink_to(archive.name)
            artifact = setup_module.inspect_openblas_prefix(
                prefix,
                step="test",
            )
            setup_module.atomic_write_json(
                cache / "provenance.json",
                {
                    "schema": setup_module.OPENBLAS_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "identity": identity,
                    "artifact": artifact,
                    "functional_probe": {
                        "linked": True,
                        "executed": True,
                    },
                },
            )
            evidence = setup_module.validate_openblas_cache(
                config,
                cache,
                expected_cache_key=cache_key,
                run_probe=False,
            )
            self.assertEqual(
                evidence["artifact"]["archive_sha256"],
                hashlib.sha256(b"static-archive").hexdigest(),
            )
            self.assertEqual(evidence["artifact"]["shared_libraries"], [])
            (prefix / "lib" / "libopenblas.so").write_bytes(b"shared")
            with self.assertRaises(setup_module.SetupContractError) as raised:
                setup_module.validate_openblas_cache(
                    config,
                    cache,
                    expected_cache_key=cache_key,
                    run_probe=False,
                )
            self.assertEqual(
                raised.exception.code,
                "openblas_shared_artifact_present",
            )

    def test_minkowski_wheel_cache_validates_fingerprint_hash_and_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
            )
            identity = {
                "repository": setup_module.MINKOWSKI_REPOSITORY,
                "commit": setup_module.MINKOWSKI_COMMIT,
                "patchset": setup_module.MINKOWSKI_PATCHSET,
            }
            cache_key = setup_module.canonical_sha256(identity)
            cache = config.build_deps_dir / "minkowski" / cache_key
            cache.mkdir(parents=True)
            wheel = cache / "minkowskiengine-0.5.4-cp311-linux_aarch64.whl"
            wheel.write_bytes(b"local-wheel")
            setup_module.atomic_write_json(
                cache / "provenance.json",
                {
                    "schema": setup_module.MINKOWSKI_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "identity": identity,
                    "patch": {
                        "patchset": setup_module.MINKOWSKI_PATCHSET,
                        "applied": True,
                    },
                    "cccl_include": "/cuda/cccl",
                    "wheel": {
                        "filename": wheel.name,
                        "sha256": hashlib.sha256(b"local-wheel").hexdigest(),
                        "size": len(b"local-wheel"),
                    },
                },
            )
            evidence = setup_module.validate_minkowski_wheel_cache(
                config,
                cache,
                expected_cache_key=cache_key,
                expected_identity=identity,
            )
            self.assertEqual(evidence["wheel"]["path"], str(wheel.resolve()))
            wheel.write_bytes(b"tampered")
            with self.assertRaises(setup_module.SetupContractError) as raised:
                setup_module.validate_minkowski_wheel_cache(
                    config,
                    cache,
                    expected_cache_key=cache_key,
                    expected_identity=identity,
                )
            self.assertEqual(
                raised.exception.code,
                "minkowski_wheel_hash_mismatch",
            )

    def test_pinned_checkout_verifies_remote_commit_and_clean_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
            )
            config.build_deps_dir.mkdir(parents=True)
            destination = config.build_deps_dir / ".staging" / "source"
            responses = [
                CompletedProcess(["clone"], 0, ""),
                CompletedProcess(["checkout"], 0, ""),
                CompletedProcess(["head"], 0, setup_module.OPENBLAS_COMMIT + "\n"),
                CompletedProcess(
                    ["remote"],
                    0,
                    setup_module.OPENBLAS_REPOSITORY + "\n",
                ),
                CompletedProcess(["status"], 0, ""),
            ]
            with mock.patch.object(
                setup_module,
                "run_command",
                side_effect=responses,
            ) as runner:
                provenance = setup_module.checkout_pinned_repository(
                    config,
                    repository=setup_module.OPENBLAS_REPOSITORY,
                    commit=setup_module.OPENBLAS_COMMIT,
                    destination=destination,
                    clone_step="openblas-clone",
                    checkout_step="openblas-checkout",
                )
            self.assertEqual(provenance["commit"], setup_module.OPENBLAS_COMMIT)
            self.assertEqual(
                provenance["repository"],
                setup_module.OPENBLAS_REPOSITORY,
            )
            clone_command = runner.call_args_list[0].args[1]
            self.assertEqual(
                clone_command,
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    setup_module.OPENBLAS_REPOSITORY,
                    str(destination),
                ],
            )
            status_command = runner.call_args_list[-1].args[1]
            self.assertIn("--untracked-files=no", status_command)

    def test_minkowski_fingerprint_covers_runtime_and_native_inputs(self):
        config = setup_module.SetupConfig(
            python_exe=sys.executable,
            ext_dir=ROOT,
            gpu_sm=121,
        )
        lane = setup_module.LANES[("linux", "aarch64", 130)]
        toolkit = {
            "cuda_home": "/usr/local/cuda-13.0",
            "release": "13.0",
        }
        openblas = {
            "cache_key": "openblas-key",
            "artifact": {"archive_sha256": "a" * 64},
        }
        python_torch = {
            "python_version": [3, 11, 9],
            "python_cache_tag": "cpython-311",
            "python_soabi": "cpython-311-aarch64-linux-gnu",
            "system": "linux",
            "machine": "aarch64",
            "torch_version": "2.9.1+cu130",
            "torch_cuda": "13.0",
        }
        identity = setup_module.minkowski_cache_identity(
            config,
            lane,
            toolkit,
            python_torch,
            openblas,
        )
        self.assertEqual(identity["gpu_sm"], 121)
        self.assertEqual(identity["patchset"], setup_module.MINKOWSKI_PATCHSET)
        self.assertEqual(identity["python_torch"], python_torch)
        self.assertEqual(
            identity["openblas"]["commit"],
            setup_module.OPENBLAS_COMMIT,
        )
        first = setup_module.canonical_sha256(identity)
        second = setup_module.canonical_sha256(dict(reversed(list(identity.items()))))
        self.assertEqual(first, second)
        changed = dict(identity)
        changed["gpu_sm"] = 120
        self.assertNotEqual(first, setup_module.canonical_sha256(changed))

    def test_atomic_promotion_never_overwrites_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            ext = self.make_extension(Path(temporary))
            config = setup_module.SetupConfig(
                python_exe=sys.executable,
                ext_dir=ext,
            )
            config.build_deps_dir.mkdir(parents=True)
            staging = config.build_deps_dir / ".staging-test"
            destination = config.build_deps_dir / "cache" / "key"
            staging.mkdir()
            (staging / "artifact").write_text("complete")
            setup_module.promote_staging_directory(
                config,
                staging,
                destination,
                step="test-promotion",
            )
            self.assertFalse(staging.exists())
            self.assertEqual((destination / "artifact").read_text(), "complete")
            another = config.build_deps_dir / ".staging-again"
            another.mkdir()
            with self.assertRaises(setup_module.SetupContractError):
                setup_module.promote_staging_directory(
                    config,
                    another,
                    destination,
                    step="test-promotion",
                )
            self.assertTrue(another.exists())

    def test_lane_selection_uses_probed_toolkit_not_driver_hint(self):
        config = setup_module.SetupConfig(
            python_exe=sys.executable,
            ext_dir=ROOT,
            cuda_version=128,
        )
        toolkits = [
            {
                "cuda_home": "/usr/local/cuda-13.0",
                "nvcc": "/usr/local/cuda-13.0/bin/nvcc",
                "release": "13.0",
                "driver_cuda_hint": 128,
            }
        ]
        with mock.patch.object(
            setup_module,
            "normalized_platform",
            return_value=("linux", "aarch64"),
        ):
            lane, toolkit = setup_module.select_lane(
                config,
                (3, 11, 9),
                toolkits,
            )
        self.assertEqual(lane.cuda_code, 130)
        self.assertEqual(lane.torch_version, "2.9.1")
        self.assertEqual(toolkit["release"], "13.0")


    def test_cuda13_nvcc_flags_merge_dedupe_and_preserve(self):
        cuda13 = setup_module.LANES[("linux", "aarch64", 130)]
        first, second = setup_module.CUDA13_NVCC_FLAGS
        existing = f"-lineinfo {first} {first} --use_fast_math --use_fast_math"
        with mock.patch.dict(
            setup_module.os.environ,
            {"NVCC_FLAGS": existing},
            clear=True,
        ):
            env = setup_module.native_build_environment(
                cuda13,
                {"cuda_home": "/usr/local/cuda-13.0"},
            )
        tokens = env["NVCC_FLAGS"].split()
        self.assertEqual(
            tokens,
            [
                "-lineinfo",
                first,
                "--use_fast_math",
                "--use_fast_math",
                second,
            ],
        )
        self.assertEqual(tokens.count(first), 1)
        self.assertEqual(tokens.count(second), 1)

        cuda12 = setup_module.LANES[("linux", "aarch64", 128)]
        with mock.patch.dict(
            setup_module.os.environ,
            {"NVCC_FLAGS": "--keep --keep"},
            clear=True,
        ):
            env = setup_module.native_build_environment(
                cuda12,
                {"cuda_home": "/usr/local/cuda-12.8"},
            )
        self.assertEqual(env["NVCC_FLAGS"], "--keep --keep")

    def test_pytorch3d_probe_exercises_quaternion_multiply(self):
        source = setup_module.RUNTIME_PROBE
        self.assertIn(
            "from pytorch3d.transforms import quaternion_apply, quaternion_multiply",
            source,
        )
        self.assertIn("quaternion_apply(quaternion, point)", source)
        self.assertIn("quaternion_multiply(quaternion, quaternion)", source)
        self.assertIn('"multiplied"', source)

    def test_minkowski_probe_places_coordinates_on_cuda_device(self):
        self.assertIn(
            "coordinates = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device)",
            setup_module.RUNTIME_PROBE,
        )

    def test_exact_cusparselt_exception_requires_rc_one_line_and_lane(self):
        cuda13 = setup_module.LANES[("linux", "aarch64", 130)]
        cuda12 = setup_module.LANES[("linux", "aarch64", 128)]
        exact = setup_module.CUSPARSELT_PIP_CHECK_OUTPUT
        self.assertTrue(
            setup_module.is_exact_cusparselt_pip_check_exception(
                cuda13,
                1,
                exact,
                "",
            )
        )
        for returncode, stdout, stderr, lane in (
            (2, exact, "", cuda13),
            (1, exact.rstrip("\n"), "", cuda13),
            (1, exact + "\n", "", cuda13),
            (1, exact + "another issue\n", "", cuda13),
            (1, exact, "warning\n", cuda13),
            (1, exact, "", cuda12),
        ):
            with self.subTest(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                lane=lane.lane_id,
            ):
                self.assertFalse(
                    setup_module.is_exact_cusparselt_pip_check_exception(
                        lane,
                        returncode,
                        stdout,
                        stderr,
                    )
                )

    def test_exact_cusparselt_exception_is_normalized_with_evidence(self):
        lane = setup_module.LANES[("linux", "aarch64", 130)]
        validation = self.successful_cusparselt_probe()
        responses = [
            CompletedProcess(["runtime"], 0, '{"ok": true}\n'),
            CompletedProcess(
                ["pip-check-capture"],
                0,
                self.canonical_probe_output({
                    "returncode": 1,
                    "stdout": setup_module.CUSPARSELT_PIP_CHECK_OUTPUT,
                    "stderr": "",
                }),
            ),
            CompletedProcess(
                ["cusparselt-probe"],
                0,
                self.canonical_probe_output(validation),
            ),
        ]
        with mock.patch.object(
            setup_module,
            "run_command",
            side_effect=responses,
        ) as runner:
            probe = setup_module.run_runtime_probe(
                setup_module.SetupConfig(
                    python_exe=sys.executable,
                    ext_dir=ROOT,
                ),
                lane,
                Path("/target/venv/bin/python"),
            )
        evidence = probe["pip_check"]
        self.assertTrue(evidence["ok"])
        self.assertFalse(evidence["pip_check_passed"])
        self.assertEqual(evidence["returncode"], 1)
        self.assertTrue(evidence["normalized_exception"]["applied"])
        self.assertEqual(
            evidence["normalized_exception"]["validation"],
            validation,
        )
        self.assertEqual(runner.call_count, 3)

    def test_pip_check_capture_with_stderr_is_not_normalized(self):
        lane = setup_module.LANES[("linux", "aarch64", 130)]
        responses = [
            CompletedProcess(["runtime"], 0, '{"ok": true}\n'),
            CompletedProcess(
                ["pip-check-capture"],
                0,
                self.canonical_probe_output({
                    "returncode": 1,
                    "stdout": setup_module.CUSPARSELT_PIP_CHECK_OUTPUT,
                    "stderr": "unexpected warning\n",
                }),
            ),
        ]
        with mock.patch.object(
            setup_module,
            "run_command",
            side_effect=responses,
        ) as runner, self.assertRaises(
            setup_module.SetupContractError,
        ) as raised:
            setup_module.run_runtime_probe(
                setup_module.SetupConfig(
                    python_exe=sys.executable,
                    ext_dir=ROOT,
                ),
                lane,
                Path("/target/venv/bin/python"),
            )
        self.assertEqual(raised.exception.code, "pip_check_failed")
        self.assertEqual(runner.call_count, 2)

    def test_cusparselt_probe_failure_remains_fail_closed(self):
        lane = setup_module.LANES[("linux", "aarch64", 130)]
        failed_validation = {
            "schema": setup_module.CUSPARSELT_PROBE_SCHEMA,
            "ok": False,
            "runtime": {
                "implementation": "cpython",
                "python_version": [3, 11, 9],
                "system": "linux",
                "machine": "aarch64",
            },
            "package": {},
            "torch": {},
            "wheel": {},
            "library": {},
            "error": "RuntimeError: RECORD sha256 mismatch",
        }
        responses = [
            CompletedProcess(["runtime"], 0, '{"ok": true}\n'),
            CompletedProcess(
                ["pip-check-capture"],
                0,
                self.canonical_probe_output({
                    "returncode": 1,
                    "stdout": setup_module.CUSPARSELT_PIP_CHECK_OUTPUT,
                    "stderr": "",
                }),
            ),
            CompletedProcess(
                ["cusparselt-probe"],
                1,
                self.canonical_probe_output(failed_validation),
            ),
        ]
        with mock.patch.object(
            setup_module,
            "run_command",
            side_effect=responses,
        ), self.assertRaises(
            setup_module.SetupContractError,
        ) as raised:
            setup_module.run_runtime_probe(
                setup_module.SetupConfig(
                    python_exe=sys.executable,
                    ext_dir=ROOT,
                ),
                lane,
                Path("/target/venv/bin/python"),
            )
        self.assertEqual(raised.exception.code, "pip_check_failed")
        evidence = raised.exception.details["probe"]["pip_check"]
        self.assertFalse(evidence["ok"])
        self.assertFalse(evidence["normalized_exception"]["applied"])
        self.assertEqual(
            evidence["normalized_exception"]["validation_failure"]["code"],
            "cusparselt_probe_failed",
        )

    def test_cusparselt_target_probe_contains_fail_closed_checks(self):
        source = setup_module.CUSPARSELT_SBSA_PROBE
        for needle in (
            'metadata.distribution(expected["package_name"])',
            'metadata.distribution("torch")',
            "stat.S_ISREG",
            "candidate.is_symlink()",
            "hashlib.file_digest",
            "header[4]",
            "header[5]",
            'struct.unpack("<H", header[18:20])',
            "ctypes.CDLL",
            "os.RTLD_NOW | os.RTLD_LOCAL",
            "loaded.cusparseLtGetProperty",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, source)


class VendoredSourceContractTests(unittest.TestCase):
    def test_upstream_metadata_and_api_symbols(self):
        upstream = json.loads((ROOT / "UPSTREAM.json").read_text())
        self.assertEqual(
            upstream["commit"],
            "0615470a5ff359c676408e9ae42036d534d04a43",
        )
        self.assertEqual(upstream["tracked_file_count"], 94)
        vendor = ROOT / "vendor" / "GaussianGPT"
        self.assertFalse((vendor / ".git").exists())
        checks = {
            vendor / "model" / "gaussian_gpt.py": [
                "def sample(",
                "num_samples=1",
                "return_lengths: bool = False",
            ],
            vendor / "utils" / "render.py": [
                "class GaussianScene:",
                "def to_dict(",
                "def render_and_save_trajectory(",
            ],
            vendor / "data" / "photoshape.py": [
                "def save_inria_ply(",
                "PlyElement.describe",
            ],
            vendor / "generate_chunks.py": [
                "GaussianVQVAE.load_from_checkpoint",
                "GaussianGPT.load_from_checkpoint",
            ],
        }
        for path, needles in checks.items():
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                with self.subTest(path=path, needle=needle):
                    self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
