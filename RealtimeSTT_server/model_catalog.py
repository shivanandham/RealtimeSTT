"""
Benchmarked models annotated for this host, shared by stt-server and stt-runner.

Each model row from model_benchmarks.json gets `installed` (its engine's package
is importable here) and `measured_on_this_host` (the row's hardware matches this
machine's CPU, or GPU for CUDA rows), so a client can grey out models that
cannot load and prefer numbers measured on similar hardware.
"""

import json
import os
import platform
import re
import subprocess
import sys
from functools import lru_cache

from RealtimeSTT.transcription_engines import engine_installed

MODEL_BENCHMARKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_benchmarks.json")


def load_benchmarks():
    """
    Returns the parsed model_benchmarks.json.
    """
    with open(MODEL_BENCHMARKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _normalize_hardware(name):
    """
    Lowercases a CPU/GPU name and drops vendor noise ("(R)", "@ 3.30GHz", "11th Gen").
    """
    name = re.sub(r"\((r|tm)\)|@.*$|\b\d+(st|nd|rd|th) gen\b|\bcpu\b", " ", (name or "").lower())
    return re.sub(r"\s+", " ", name).strip()


@lru_cache(maxsize=1)
def host_hardware():
    """
    Returns {"cpu": brand, "gpu": CUDA device name or None} for this machine.
    """
    cpu = None
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), None)
    except (OSError, subprocess.SubprocessError):
        cpu = None
    gpu = None
    torch = sys.modules.get("torch")  # Only when already loaded; importing torch here costs seconds.
    try:
        if torch is not None and torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = None
    return {"cpu": cpu or platform.processor() or None, "gpu": gpu}


def _measured_on_this_host(row, benchmark, host):
    """
    Reports whether a row was measured on hardware matching this machine.
    """
    gpu_row = row.get("device") == "cuda"
    measured = row.get("hardware") or (benchmark.get("hardware") or {}).get("gpu" if gpu_row else "cpu")
    mine = _normalize_hardware(host["gpu"] if gpu_row else host["cpu"])
    return bool(mine) and mine in _normalize_hardware(measured)


def model_catalog():
    """
    Returns {"benchmark", "models", "host"} with every model row annotated for this host.
    """
    benchmarks = load_benchmarks()
    host = host_hardware()
    models = []
    for row in benchmarks["models"]:
        try:
            installed = engine_installed(row["engine"])
        except Exception:
            installed = False
        models.append({**row, "installed": installed,
                       "measured_on_this_host": _measured_on_this_host(row, benchmarks["benchmark"], host)})
    return {"benchmark": benchmarks["benchmark"], "models": models, "host": host}
