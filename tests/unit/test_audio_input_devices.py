import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from RealtimeSTT.core import audio_input_worker as worker


class FakeInterface:
    def __init__(self, devices):
        self.devices = devices

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        name, inputs = self.devices[index]
        return {"name": name, "maxInputChannels": inputs, "defaultSampleRate": 48000.0}


class DeviceLookupTests(unittest.TestCase):
    def setUp(self):
        self.interface = FakeInterface([
            ("HDMI out", 0),
            ("USB PnP Audio Device: Audio (hw:1,0)", 1),
            ("HDA Intel PCH: ALC294 Analog (hw:2,0)", 2),
            ("pulse", 32),
            ("default", 32),
        ])

    def test_list_input_devices_skips_outputs(self):
        devices = worker.list_input_devices(self.interface)
        self.assertEqual([d["index"] for d in devices], [1, 2, 3, 4])
        self.assertEqual(devices[0]["name"], "USB PnP Audio Device: Audio (hw:1,0)")
        self.assertEqual(devices[0]["default_sample_rate"], 48000)

    def test_find_by_name_prefers_exact_then_substring(self):
        self.assertEqual(worker._find_input_device_by_name(self.interface, ("pulse", "default")), 3)
        self.assertEqual(worker._find_input_device_by_name(self.interface, ("default",)), 4)
        self.assertIsNone(worker._find_input_device_by_name(self.interface, ("usb pnp",)))
        self.assertEqual(worker._find_input_device_by_name(self.interface, ("usb pnp",), substring=True), 1)
        self.assertEqual(worker._find_input_device_by_name(self.interface, ("alc294",), substring=True), 2)
        self.assertIsNone(worker._find_input_device_by_name(self.interface, ("HDMI",), substring=True))


class PulseSourceListTests(unittest.TestCase):
    def test_parses_pactl_json_and_flags_monitors(self):
        payload = json.dumps([
            {"name": "alsa_input.usb-mic.mono-fallback", "description": "USB Mic"},
            {"name": "alsa_output.pci.analog-stereo.monitor", "description": "Monitor of Speakers"},
            {"index": 3},
        ])
        with patch("subprocess.run", return_value=SimpleNamespace(stdout=payload)):
            sources = worker.list_pulse_sources()
        self.assertEqual(sources, [
            {"name": "alsa_input.usb-mic.mono-fallback", "description": "USB Mic", "monitor": False},
            {"name": "alsa_output.pci.analog-stereo.monitor", "description": "Monitor of Speakers", "monitor": True},
        ])

    def test_missing_pactl_gives_empty_list(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("pactl")):
            self.assertEqual(worker.list_pulse_sources(), [])


class SharedInputInfoTests(unittest.TestCase):
    def test_round_trip(self):
        info = worker.make_input_device_info()
        self.assertIsNone(worker.read_input_device_info(info))
        self.assertIsNone(worker.read_input_device_info(None))
        info["name"].value = "USB PnP Audio Device".encode("utf-8")
        info["sample_rate"].value = 48000
        info["index"].value = 4
        self.assertEqual(worker.read_input_device_info(info),
                         {"index": 4, "name": "USB PnP Audio Device", "sample_rate": 48000})


class PulseSourceEnvTests(unittest.TestCase):
    def test_sets_and_restores_pulse_source(self):
        os.environ.pop("PULSE_SOURCE", None)
        with worker._pulse_source_env("alsa_output.x.monitor"):
            self.assertEqual(os.environ["PULSE_SOURCE"], "alsa_output.x.monitor")
        self.assertNotIn("PULSE_SOURCE", os.environ)
        os.environ["PULSE_SOURCE"] = "previous"
        try:
            with worker._pulse_source_env("other"):
                self.assertEqual(os.environ["PULSE_SOURCE"], "other")
            self.assertEqual(os.environ["PULSE_SOURCE"], "previous")
            with worker._pulse_source_env(None):
                self.assertEqual(os.environ["PULSE_SOURCE"], "previous")
        finally:
            os.environ.pop("PULSE_SOURCE", None)


if __name__ == "__main__":
    unittest.main()
