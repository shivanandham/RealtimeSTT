import copy
import json
import threading
import time
import unittest
from unittest.mock import patch

from RealtimeSTT.core.model_defaults import apply_device_defaults
from RealtimeSTT.multichannel import MultiChannelRecorder, SharedEngine
from RealtimeSTT.transcription_engines import TranscriptionResult
from RealtimeSTT_server import stt_runner
from RealtimeSTT_server.stt_runner import EngineSwitcher, Runner, build_parser, engine_config

SHERPA_V2 = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


class FakeEngine:
    def __init__(self, name, config):
        self.name, self.config = name, config

    def transcribe(self, audio, language=None, use_prompt=True):
        prompt = self.config.initial_prompt if use_prompt else None
        return TranscriptionResult(text=f"{self.name}:{self.config.model}:{prompt}")


def fake_factory(name, config):
    if config.model == "broken":
        raise RuntimeError("cannot load broken")
    return FakeEngine(name, config)


def shared(name, config):
    return SharedEngine(name, config, warmup=False, engine_factory=fake_factory)


class EngineSwitcherTests(unittest.TestCase):
    def setUp(self):
        # sherpa-onnx "installed" so the CPU default is Parakeet v2 serving both lanes.
        find_spec = patch("importlib.util.find_spec",
                          side_effect=lambda name: object() if name == "sherpa_onnx" else None)
        installed = patch.object(stt_runner, "engine_installed", side_effect=lambda name: name != "qwen3_asr")
        for patcher in (find_spec, installed):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.events = []
        self.event_added = threading.Condition()

    def write(self, event):
        json.dumps(event)  # every event must be serializable
        with self.event_added:
            self.events.append(event)
            self.event_added.notify_all()

    def start(self, *argv):
        args = build_parser().parse_args(["--device", "cpu", *argv])
        base = copy.copy(args)
        apply_device_defaults(args)
        final = shared(args.engine, engine_config(args)).wait_ready()
        if args.no_realtime:
            realtime = None
        elif args.use_main_model_for_realtime:
            realtime = final
        else:
            realtime = shared(args.rt_engine or args.engine, engine_config(args, realtime=True)).wait_ready()
        switcher = EngineSwitcher(base, args, self.write, engine_factory=shared)
        multi = MultiChannelRecorder([], self.write, final, realtime, recorder_factory=lambda **kw: None)
        switcher.recorder = multi
        self.addCleanup(multi.shutdown)
        self.addCleanup(switcher.close)
        return switcher, multi

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        with self.event_added:
            while True:
                for event in self.events:
                    if predicate(event):
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail(f"event not seen: {self.events}")
                self.event_added.wait(remaining)

    def run_command(self, switcher, request_id, **command):
        switcher.submit({"request_id": request_id, **command})
        return self.wait_for(lambda e: e.get("request_id") == request_id and e["type"] in ("error", "engine")
                             and e.get("status") != "loading")

    def test_startup_choices_defaults_and_active_block(self):
        switcher, multi = self.start("--hotwords", "JSON, SQL")
        active = switcher.active()
        self.assertEqual((active["transcription_engine"], active["model"]), ("sherpa_onnx_parakeet", SHERPA_V2))
        self.assertTrue(active["use_main_model_for_realtime"])
        self.assertEqual(active["choices"], {"final": {"mode": "default"}, "realtime": {"mode": "default"}})
        self.assertEqual(active["defaults"], {"final": {"engine": "sherpa_onnx_parakeet", "model": SHERPA_V2},
                                              "realtime": {"same_as_final": True}})
        self.assertEqual((active["hotwords"], active["device"], active["loading"]), ("JSON, SQL", "cpu", {}))
        self.assertEqual(multi.final_engine.config.engine_options, {"hotwords": "JSON, SQL"})

    def test_switching_final_reresolves_default_realtime_and_back(self):
        switcher, multi = self.start()
        parakeet = multi.final_engine

        reply = self.run_command(switcher, 1, command="set_engine", lane="final", engine="faster_whisper",
                                 model="small.en")
        loading = self.wait_for(lambda e: e.get("request_id") == 1 and e.get("status") == "loading")
        self.assertEqual(loading["active"]["loading"], {"final": {"engine": "faster_whisper", "model": "small.en"},
                                                        "realtime": {"engine": "faster_whisper", "model": "tiny.en"}})
        # The old engine served until the swap.
        self.assertEqual(loading["active"]["model"], SHERPA_V2)
        self.assertEqual((reply["status"], reply["lanes"]), ("ready", ["final", "realtime"]))
        self.assertEqual((reply["active"]["model"], reply["active"]["realtime_model"]), ("small.en", "tiny.en"))
        self.assertEqual(reply["active"]["defaults"]["realtime"], {"engine": "faster_whisper", "model": "tiny.en"})
        self.assertEqual(multi.final_engine.transcribe("a").text, "faster_whisper:small.en:None")
        parakeet._thread.join(5)
        self.assertFalse(parakeet._thread.is_alive(), "the replaced engine is closed")

        reply = self.run_command(switcher, 2, command="set_engine", lane="realtime", same_as_final=True)
        self.assertIs(multi.realtime_engine, multi.final_engine)
        self.assertEqual(reply["active"]["choices"]["realtime"], {"mode": "same_as_final"})

        reply = self.run_command(switcher, 3, command="set_engine", lane="final", default=True)
        self.assertEqual(reply["active"]["model"], SHERPA_V2)
        self.assertIs(multi.realtime_engine, multi.final_engine)

    def test_explicit_realtime_keeps_final_and_load_errors_keep_old_engines(self):
        switcher, multi = self.start()
        final = multi.final_engine
        reply = self.run_command(switcher, 1, command="set_engine", lane="realtime", engine="faster_whisper",
                                 model="base.en")
        self.assertEqual(reply["lanes"], ["realtime"])
        self.assertIs(multi.final_engine, final)
        self.assertEqual(multi.realtime_engine.config.model, "base.en")

        error = self.run_command(switcher, 2, command="set_engine", lane="final", model="broken")
        self.assertEqual((error["type"], error["command"]), ("error", "set_engine"))
        self.assertIn("cannot load broken", error["message"])
        self.assertEqual(error["active"]["model"], SHERPA_V2)
        self.assertIs(multi.final_engine, final)
        self.assertEqual(switcher.active()["loading"], {})
        self.assertEqual(switcher.choices["final"], {"mode": "default"})

    def test_prompt_changes_apply_in_place_and_hotwords_reload(self):
        switcher, multi = self.start("-m", "small.en", "-r", "tiny.en")
        final, realtime = multi.final_engine, multi.realtime_engine
        reply = self.run_command(switcher, 1, command="set_initial_prompt", initial_prompt="JSON, SQL")
        self.assertEqual(reply["lanes"], [])
        self.assertIs(multi.final_engine, final)
        self.assertEqual(final.transcribe("a").text, "faster_whisper:small.en:JSON, SQL")
        # The separate realtime model falls back to the final prompt.
        self.assertEqual(realtime.transcribe("a").text, "faster_whisper:tiny.en:JSON, SQL")
        self.run_command(switcher, 2, command="set_initial_prompt", lane="realtime", initial_prompt="RT")
        self.assertEqual(realtime.transcribe("a").text, "faster_whisper:tiny.en:RT")

        reply = self.run_command(switcher, 3, command="set_engine", lane="final", engine="sherpa_onnx_parakeet",
                                 model=SHERPA_V2)
        self.assertEqual(reply["lanes"], ["final"])  # realtime stays the explicit tiny.en
        reply = self.run_command(switcher, 4, command="set_hotwords", hotwords="Kubernetes, Acme")
        self.assertEqual(reply["lanes"], ["final"])
        self.assertEqual(multi.final_engine.config.engine_options, {"hotwords": "Kubernetes, Acme"})
        self.assertEqual(reply["active"]["hotwords"], "Kubernetes, Acme")
        self.assertIs(multi.realtime_engine, realtime)

    def test_invalid_requests_are_rejected_immediately(self):
        switcher, _ = self.start()
        for command, message in (
            ({"command": "set_engine", "lane": "both", "model": "x"}, "lane"),
            ({"command": "set_engine", "lane": "final", "same_as_final": True}, "realtime lane"),
            ({"command": "set_engine", "lane": "final"}, "needs engine"),
            ({"command": "set_engine", "lane": "final", "engine": "qwen3_asr"}, "not installed"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                switcher.submit(command)

    def test_realtime_cannot_be_switched_when_disabled(self):
        switcher, _ = self.start("--no-realtime")
        self.assertEqual(switcher.active()["choices"]["realtime"], {"mode": "off"})
        with self.assertRaisesRegex(ValueError, "disabled"):
            switcher.submit({"command": "set_engine", "lane": "realtime", "model": "tiny.en"})

    def test_runner_routes_commands_and_adds_catalog_to_info(self):
        switcher, multi = self.start()
        runner = Runner(multi, self.write, switcher=switcher)
        with patch.object(stt_runner, "list_pulse_sources", return_value=[]), \
                patch.object(Runner, "list_devices", return_value=[]):
            info = runner.info()
        self.assertEqual(info["active"]["model"], SHERPA_V2)
        self.assertIn("benchmark", info)
        self.assertTrue(all("installed" in m and "measured_on_this_host" in m for m in info["models"]))
        self.assertIn("cpu", info["host"])
        self.assertIn("set_engine", info["protocol"]["stdin"])
        runner.handle(json.dumps({"command": "set_engine", "lane": "nope", "request_id": 9}))
        self.assertEqual(self.wait_for(lambda e: e.get("request_id") == 9)["type"], "error")


if __name__ == "__main__":
    unittest.main()
