import argparse
import unittest
from unittest.mock import patch

from RealtimeSTT_server import stt_server
from RealtimeSTT_server.stt_server import apply_device_defaults


with_nemo = patch("importlib.util.find_spec", return_value=object())
no_nemo = patch("importlib.util.find_spec", return_value=None)


def args(**overrides):
    values = dict(device="auto", model=None, rt_model=None, compute_type=None, engine=None, rt_engine=None,
                  lang="en", use_main_model_for_realtime=None)
    values.update(overrides)
    return argparse.Namespace(**values)


class SttServerDefaultsTests(unittest.TestCase):
    def test_auto_uses_gpu_defaults_when_cuda_is_available(self):
        with patch("torch.cuda.is_available", return_value=True), no_nemo:
            resolved = apply_device_defaults(args())
        self.assertEqual(
            (resolved.device, resolved.model, resolved.rt_model, resolved.compute_type),
            ("cuda", "large-v3-turbo", "small.en", "float16"),
        )

    def test_auto_falls_back_to_cpu_defaults(self):
        with patch("torch.cuda.is_available", return_value=False):
            resolved = apply_device_defaults(args())
        self.assertEqual(
            (resolved.device, resolved.model, resolved.rt_model, resolved.compute_type),
            ("cpu", "small", "tiny.en", "int8"),
        )

    def test_gpu_with_nemo_uses_parakeet_for_final_and_realtime(self):
        with patch("torch.cuda.is_available", return_value=True), with_nemo:
            resolved = apply_device_defaults(args())
        self.assertEqual(
            (resolved.engine, resolved.model, resolved.use_main_model_for_realtime, resolved.rt_engine),
            ("parakeet", "nvidia/parakeet-tdt-0.6b-v2", True, None),
        )

    def test_parakeet_is_skipped_for_non_english_or_cpu(self):
        with with_nemo:
            german = apply_device_defaults(args(device="cuda", lang="de"))
            cpu = apply_device_defaults(args(device="cpu"))
        self.assertEqual((german.engine, german.model), ("faster_whisper", "large-v3-turbo"))
        self.assertEqual((cpu.engine, cpu.model), ("faster_whisper", "small"))

    def test_non_whisper_engine_gets_whisper_realtime_engine(self):
        with no_nemo:
            resolved = apply_device_defaults(args(device="cuda", engine="qwen3_asr"))
        self.assertEqual(
            (resolved.model, resolved.rt_engine, resolved.rt_model),
            (None, "faster_whisper", "small.en"),
        )

    def test_explicit_values_are_kept(self):
        resolved = apply_device_defaults(args(device="cpu", model="medium.en", compute_type="float32"))
        self.assertEqual(
            (resolved.device, resolved.model, resolved.rt_model, resolved.compute_type),
            ("cpu", "medium.en", "tiny.en", "float32"),
        )


class SttServerInfoTests(unittest.TestCase):
    def test_get_info_describes_active_setup_input_and_models(self):
        recorder = argparse.Namespace(transcription_engine="faster_whisper", realtime_transcription_engine=None,
                                      device="cuda", language="en")
        config = {"model": "large-v3-turbo", "realtime_model_type": "small.en", "compute_type": "float16",
                  "enable_realtime_transcription": True, "on_recording_start": print}
        client = {"remote_address": "127.0.0.1:5000", "sample_rate": 48000,
                  "input_device": {"index": 4, "name": "USB mic"}}
        with patch.object(stt_server, "recorder", recorder), \
                patch.object(stt_server, "recorder_config", config), \
                patch.object(stt_server, "global_args", argparse.Namespace(control=8011, data=8012)), \
                patch.object(stt_server, "data_clients", {"ws": client}):
            info = stt_server.build_server_info()

        self.assertEqual(info["active"]["model"], "large-v3-turbo")
        self.assertEqual(info["active"]["realtime_transcription_engine"], "faster_whisper")
        self.assertEqual(info["active"]["device"], "cuda")
        self.assertEqual(info["input"]["clients"], [client])
        self.assertNotIn("on_recording_start", info["config"])
        self.assertTrue(any(m.get("default_for") == ["cuda:final"] for m in info["models"]))
        self.assertIn("hardware", info["benchmark"])
        self.assertIn("get_info", info["protocol"]["control_commands"])
        self.assertIn("language", info["protocol"]["allowed_parameters"])
        self.assertIn("fullSentence", info["protocol"]["data_messages"]["types"])


if __name__ == "__main__":
    unittest.main()
