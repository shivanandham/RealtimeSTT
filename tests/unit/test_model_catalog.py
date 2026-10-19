import unittest
from unittest.mock import patch

from RealtimeSTT_server import model_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_rows_are_marked_installed_and_matched_to_this_host(self):
        host = {"cpu": "11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz", "gpu": "NVIDIA GeForce RTX 3070 Laptop GPU"}
        benchmarks = {
            "benchmark": {"hardware": {"cpu": "Intel Core i7-11370H (8 threads)",
                                       "gpu": "NVIDIA GeForce RTX 3070 Laptop GPU (8 GB)"}},
            "models": [
                {"engine": "faster_whisper", "model": "tiny.en", "device": "cpu"},
                {"engine": "parakeet", "model": "p", "device": "cuda"},
                {"engine": "sherpa_onnx_parakeet", "model": "s", "device": "cpu", "hardware": "Apple M4 (16 GB)"},
                {"engine": "not_an_engine", "model": "x", "device": "cpu"},
            ],
        }
        with patch.object(model_catalog, "load_benchmarks", return_value=benchmarks), \
                patch.object(model_catalog, "host_hardware", return_value=host), \
                patch.object(model_catalog, "engine_installed", side_effect=lambda e: e == "faster_whisper"
                             or (e == "not_an_engine" and 1 / 0)):
            rows = model_catalog.model_catalog()["models"]
        self.assertEqual([(r["installed"], r["measured_on_this_host"]) for r in rows],
                         [(True, True), (False, True), (False, False), (False, True)])

    def test_apple_chip_variants_do_not_match_each_other(self):
        self.assertIn(model_catalog._normalize_hardware("Apple M4"), model_catalog._normalize_hardware("Apple M4 (4 performance)"))
        self.assertNotIn(model_catalog._normalize_hardware("Apple M4 Pro"), model_catalog._normalize_hardware("Apple M4 (16 GB)"))


if __name__ == "__main__":
    unittest.main()
