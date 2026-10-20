import queue
import shlex
import struct
import sys
import threading
import time
import unittest
from types import SimpleNamespace

from RealtimeSTT.multichannel import ChannelSpec, MultiChannelRecorder, SharedEngine, pcm_to_int16
from RealtimeSTT.transcription_engines import TranscriptionEngineConfig, TranscriptionResult


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def warmup(self, audio):
        self.calls.append(("warmup", None))

    def transcribe(self, audio, language=None, use_prompt=True):
        self.entered.set()
        self.gate.wait()
        self.calls.append((audio, language))
        return TranscriptionResult(text=f"heard {audio}")


def make_shared(engine, warmup=False):
    shared = SharedEngine(
        "fake", TranscriptionEngineConfig(model="m"), warmup=warmup, engine_factory=lambda name, config: engine
    )
    shared.wait_ready(timeout=5)
    return shared


class SharedEngineTests(unittest.TestCase):
    def test_lanes_transcribe_and_record_stats(self):
        engine = FakeEngine()
        shared = make_shared(engine)
        result = shared.lane("realtime").transcribe("a1", language="en")
        self.assertEqual(result.text, "heard a1")
        self.assertEqual(engine.calls, [("a1", "en")])
        info = shared.info()
        self.assertEqual(info["lanes"]["realtime"]["calls"], 1)
        self.assertEqual(info["lanes"]["final"]["calls"], 0)
        self.assertEqual(info["model"], "m")
        shared.close()
        self.assertFalse(shared._thread.is_alive())

    def test_final_requests_run_before_queued_realtime(self):
        engine = FakeEngine()
        shared = make_shared(engine)
        engine.gate.clear()
        threads = [threading.Thread(target=shared.transcribe, args=("block",), kwargs={"kind": "realtime"})]
        threads[0].start()
        self.assertTrue(engine.entered.wait(2))
        for audio, kind in (("r1", "realtime"), ("f1", "final")):
            threads.append(threading.Thread(target=shared.transcribe, args=(audio,), kwargs={"kind": kind}))
            threads[-1].start()
        time.sleep(0.1)  # let both land in the queue behind the blocked job
        engine.gate.set()
        for thread in threads:
            thread.join(2)
        self.assertEqual([audio for audio, _ in engine.calls], ["block", "f1", "r1"])
        shared.close()

    def test_load_error_surfaces(self):
        def broken(name, config):
            raise RuntimeError("no nemo")

        shared = SharedEngine("fake", TranscriptionEngineConfig(model="m"), warmup=False, engine_factory=broken)
        with self.assertRaises(RuntimeError):
            shared.wait_ready(timeout=5)
        with self.assertRaises(RuntimeError):
            shared.transcribe("x")
        self.assertEqual(shared.info()["error"], "no nemo")
        shared.close()


class FakeRecorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_running = True
        self.is_shut_down = False
        self.is_recording = False
        self.use_microphone = SimpleNamespace(value=True)
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.input_device_info = {"index": kwargs.get("input_device_index"), "name": "fake", "sample_rate": 16000}
        self.texts = queue.Queue()
        self.calls = []

    def text(self):
        return self.texts.get() or ""

    def set_microphone(self, on):
        self.calls.append(("mic", on))

    def clear_audio_queue(self):
        self.calls.append(("clear",))

    def feed_audio(self, chunk):
        self.calls.append(("feed", chunk))

    def shutdown(self):
        self.is_shut_down = True
        self.is_running = False
        self.texts.put(None)

    def speak(self, text, start=1.0, stop=2.0):
        cb = self.kwargs
        self.recording_start_time = start
        self.is_recording = True
        cb["on_recording_start"]()
        cb["on_realtime_transcription_update"]("partial")
        cb["on_realtime_transcription_stabilized"]("partial")
        cb["on_turn_detection_start"]()
        cb["on_turn_detection_stop"]()  # speech continued
        self.is_recording = False
        self.recording_stop_time = stop
        cb["on_recording_stop"]()
        cb["on_turn_detection_stop"]()  # after the stop: not a resume
        self.assertion = cb["on_transcription_start"]([0.0] * 16000)
        self.last_recording_start_time = start
        self.last_recording_stop_time = stop
        self.texts.put(text)


class MultiChannelRecorderTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.shared = make_shared(self.engine)
        self.events = []
        self.recorders = []
        self.addCleanup(self.shared.close)

    def factory(self, **kwargs):
        recorder = FakeRecorder(**kwargs)
        self.recorders.append(recorder)
        return recorder

    def make(self, realtime=True, **kwargs):
        multi = MultiChannelRecorder(
            [ChannelSpec("me", input_device_index=4), ChannelSpec("them", pulse_source="mon")],
            on_event=self.events.append,
            final_engine=self.shared,
            realtime_engine=self.shared if realtime else None,
            recorder_factory=self.factory,
            language="en",
            **kwargs,
        )
        self.addCleanup(multi.shutdown)
        return multi

    def wait_for(self, type_, channel, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.events:
                if event["type"] == type_ and event["channel"] == channel:
                    return event
            time.sleep(0.01)
        self.fail(f"no {type_} event for {channel}: {self.events}")

    def test_recorders_share_engine_lanes(self):
        self.make(post_speech_silence_duration=0.3)
        me, them = self.recorders
        self.assertEqual(me.kwargs["input_device_index"], 4)
        self.assertIsNone(me.kwargs["pulse_source"])
        self.assertEqual(them.kwargs["pulse_source"], "mon")
        for recorder in (me, them):
            self.assertTrue(recorder.kwargs["use_microphone"])
            self.assertFalse(recorder.kwargs["spinner"])
            self.assertFalse(recorder.kwargs["use_main_model_for_realtime"])
            self.assertTrue(recorder.kwargs["enable_realtime_transcription"])
            self.assertEqual(recorder.kwargs["language"], "en")
            self.assertEqual(recorder.kwargs["post_speech_silence_duration"], 0.3)
            final, realtime = recorder.kwargs["transcription_executor"], recorder.kwargs["realtime_transcription_executor"]
            self.assertIs(final.engine, self.shared)
            self.assertEqual((final.kind, realtime.kind), ("final", "realtime"))
            self.assertEqual(final.transcribe("x").text, "heard x")

    def test_events_are_tagged_with_channel_and_utterance_id(self):
        self.make()
        _, them = self.recorders
        them.speak("Hello there.")
        final = self.wait_for("fullSentence", "them")
        types = [e["type"] for e in self.events if e["channel"] == "them"]
        self.assertEqual(types, [
            "speech_start", "realtime", "realtime_stabilized", "speech_paused", "speech_resumed",
            "speech_end", "transcription_start", "fullSentence",
        ])
        self.assertTrue(all(e["utterance_id"] == 1 for e in self.events))
        self.assertEqual(self.events[0]["speech_start_ms"], 1000)
        self.assertEqual(self.wait_for("speech_end", "them")["speech_end_ms"], 2000)
        self.assertEqual(self.wait_for("transcription_start", "them")["audio_s"], 1.0)
        self.assertEqual(final["text"], "Hello there.")
        self.assertGreaterEqual(final["latency_ms"], 0)
        self.assertFalse(them.assertion, "on_transcription_start must not abort transcription")
        self.assertFalse([e for e in self.events if e["channel"] == "me"])
        them.speak("Second one.")
        self.events.clear()
        self.assertEqual(self.wait_for("fullSentence", "them")["utterance_id"], 2)

    def test_realtime_disabled_without_realtime_engine(self):
        self.make(realtime=False)
        for recorder in self.recorders:
            self.assertFalse(recorder.kwargs["enable_realtime_transcription"])
            self.assertNotIn("realtime_transcription_executor", recorder.kwargs)

    def test_controls_target_one_or_all_channels(self):
        multi = self.make()
        me, them = self.recorders
        multi.pause("me")
        multi.resume()
        multi.clear("them")
        multi.set_parameter("post_speech_silence_duration", 0.45, "me")
        self.assertEqual(me.calls, [("mic", False), ("mic", True)])
        self.assertEqual(them.calls, [("mic", True), ("clear",)])
        self.assertEqual(me.post_speech_silence_duration, 0.45)
        self.assertFalse(hasattr(them, "post_speech_silence_duration"))
        with self.assertRaises(KeyError):
            multi.pause("nope")

    def test_set_channel_replaces_only_that_channel(self):
        multi = self.make()
        old_me, them = self.recorders
        multi.set_channel(ChannelSpec("me", input_device_index=5))
        new_me = self.recorders[-1]
        self.assertTrue(old_me.is_shut_down)
        self.assertFalse(them.is_shut_down)
        self.assertEqual(new_me.kwargs["input_device_index"], 5)
        self.assertEqual(sorted(multi.channel_names()), ["me", "them"])
        old_me.kwargs["on_recording_start"]()  # a straggler from the stopped recorder
        self.assertEqual(self.events, [])
        self.assertTrue(multi.remove_channel("them"))
        self.assertTrue(them.is_shut_down)
        self.assertFalse(multi.remove_channel("them"))
        self.assertEqual(multi.channel_names(), ["me"])

    def test_info_and_wait_for_inputs(self):
        multi = self.make()
        info = multi.info()
        self.assertEqual(info["channels"]["me"]["input_device"]["index"], 4)
        self.assertEqual(info["channels"]["them"]["pulse_source"], "mon")
        self.assertFalse(info["channels"]["me"]["paused"])
        self.assertTrue(info["engines"]["realtime_shares_final"])
        self.assertEqual(info["engines"]["final"]["engine"], "fake")
        self.assertEqual(set(multi.wait_for_inputs(1)), {"me", "them"})

    def test_shutdown_stops_channels_and_distinct_engines(self):
        other = make_shared(FakeEngine())
        multi = MultiChannelRecorder(
            [ChannelSpec("me")], on_event=self.events.append, final_engine=self.shared,
            realtime_engine=other, recorder_factory=self.factory,
        )
        multi.shutdown()
        self.assertTrue(self.recorders[0].is_shut_down)
        self.assertFalse(other._thread.is_alive())
        self.assertFalse(self.shared._thread.is_alive())
        self.assertEqual(multi.channel_names(), [])


def python_command(code):
    """
    Returns a command line running code with this interpreter.
    """
    return shlex.join([sys.executable, "-c", code])


class CommandChannelTests(unittest.TestCase):
    def setUp(self):
        self.shared = make_shared(FakeEngine())
        self.addCleanup(self.shared.close)
        self.events = []
        self.recorders = []

    def factory(self, **kwargs):
        recorder = FakeRecorder(**kwargs)
        self.recorders.append(recorder)
        return recorder

    def make(self, *specs):
        multi = MultiChannelRecorder(list(specs), on_event=self.events.append, final_engine=self.shared,
                                     realtime_engine=self.shared, recorder_factory=self.factory)
        self.addCleanup(multi.shutdown)
        return multi

    def wait_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail(f"condition not met; events: {self.events}")

    def fed(self, recorder):
        return b"".join(chunk for call, *rest in recorder.calls if call == "feed" for chunk in rest)

    def test_pcm_conversion(self):
        mono = struct.pack("<4h", 1, -2, 300, -32768)
        self.assertEqual(pcm_to_int16(mono), mono)
        stereo_float = struct.pack("<4f", 0.5, 0.5, -1.0, 0.0)
        self.assertEqual(struct.unpack("<2h", pcm_to_int16(stereo_float, "f32le", channels=2)), (16383, -16383))
        self.assertEqual(len(pcm_to_int16(b"\0\0" * 480, rate=48000)), 2 * 160)

    def test_command_stdout_is_fed_and_exit_is_reported(self):
        code = ("import sys; sys.stdout.buffer.write(bytes(range(1, 11))); sys.stdout.flush();"
                "sys.stderr.write('tap: started\\nno permission\\n'); sys.exit(3)")
        multi = self.make(ChannelSpec("them", command=python_command(code)))
        recorder = self.recorders[0]
        self.assertFalse(recorder.kwargs["use_microphone"])
        error = None
        self.wait_until(lambda: any(e["type"] == "error" for e in self.events))
        error = next(e for e in self.events if e["type"] == "error")
        self.assertEqual((error["channel"], error["exit_code"]), ("them", 3))
        self.assertEqual(error["stderr"], ["tap: started", "no permission"])
        self.assertIn("exited with code 3: no permission", error["message"])
        warnings = [e["message"] for e in self.events if e["type"] == "warning"]
        self.assertEqual(warnings, ["tap: started", "no permission"])
        self.assertEqual(self.fed(recorder), bytes(range(1, 11)))
        info = multi.info()["channels"]["them"]
        self.assertEqual((info["input_device"]["source"], info["input_device"]["exit_code"]), ("command", 3))
        self.assertEqual(info["command"], python_command(code))

    def test_pause_drops_audio_and_remove_stops_the_process(self):
        code = "import sys, time\nwhile True:\n    sys.stdout.buffer.write(b'\\1\\0' * 160); sys.stdout.flush(); time.sleep(0.01)"
        multi = self.make(ChannelSpec("them", command=python_command(code)))
        recorder = self.recorders[0]
        source = multi._channels["them"].source
        self.wait_until(lambda: self.fed(recorder))
        self.assertEqual(multi.wait_for_inputs(1)["them"]["pid"], source.process.pid)
        multi.pause("them")
        self.assertTrue(multi.info()["channels"]["them"]["paused"])
        time.sleep(0.05)
        before = len(self.fed(recorder))
        time.sleep(0.15)
        self.assertEqual(len(self.fed(recorder)), before)
        self.assertNotIn(("mic", False), recorder.calls)
        multi.resume("them")
        self.wait_until(lambda: len(self.fed(recorder)) > before)
        self.assertTrue(multi.remove_channel("them"))
        self.assertIsNotNone(source.process.poll())
        self.assertTrue(recorder.is_shut_down)
        self.assertFalse(any(e["type"] == "error" for e in self.events), "a requested stop is not an error")

    def test_invalid_command_specs_keep_the_running_channel(self):
        multi = self.make(ChannelSpec("them", input_device_index=4))
        for spec, message in (
            (ChannelSpec("them", command="definitely-not-a-command-xyz"), "not found"),
            (ChannelSpec("them", command=python_command("pass"), pcm_format="u8"), "pcm_format"),
            (ChannelSpec("them", command=python_command("pass"), pcm_rate=0), "positive"),
            (ChannelSpec("them", 4, command=python_command("pass")), "cannot be combined"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                multi.set_channel(spec)
        self.assertEqual(len(self.recorders), 1)
        self.assertFalse(self.recorders[0].is_shut_down)


if __name__ == "__main__":
    unittest.main()
