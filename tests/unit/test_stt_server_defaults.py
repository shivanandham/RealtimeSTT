import argparse
import unittest
from unittest.mock import patch

from RealtimeSTT_server import stt_server
from RealtimeSTT_server.stt_server import apply_device_defaults


def args(**overrides):
    values = dict(device="auto", model=None, rt_model=None, compute_type=None)
    values.update(overrides)
    return argparse.Namespace(**values)


class SttServerDefaultsTests(unittest.TestCase):
    def test_auto_uses_gpu_defaults_when_cuda_is_available(self):
        with patch("torch.cuda.is_available", return_value=True):
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


if __name__ == "__main__":
    unittest.main()
