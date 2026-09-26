#!/usr/bin/env python3
"""Run repeatable LiteRT GPU benchmark sessions through ADB.

The Android debug APK must already be built. This controller installs it,
copies each TFLite model into app-private storage, starts a fresh process for
each session, retrieves JSON results, captures logcat, and appends delegate
coverage parsed from TensorFlow Lite GPU-delegate messages.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE = "org.lightmis.mobilebenchmark"
ACTIVITY = f"{PACKAGE}/.BenchmarkActivity"
DELEGATE_RE = re.compile(
    r"Replacing\s+(\d+)\s+out of\s+(\d+)\s+node\(s\).*?yielding\s+(\d+)\s+partitions?",
    flags=re.IGNORECASE,
)


def run_adb(
    adb: str,
    arguments: list[str],
    *,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [adb, *arguments]
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def safe_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot derive a safe model name from {path.name!r}")
    return cleaned


def ensure_one_device(adb: str) -> None:
    output = run_adb(adb, ["devices"]).stdout.splitlines()[1:]
    devices = [line.split()[0] for line in output if line.strip().endswith("\tdevice")]
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one authorized ADB device, found {devices}")


def getprop(adb: str, key: str) -> str:
    return run_adb(adb, ["shell", "getprop", key]).stdout.strip()


def shell_output(adb: str, command: str) -> str:
    return run_adb(adb, ["shell", command]).stdout.strip()


def collect_device_info(adb: str) -> dict[str, Any]:
    meminfo = shell_output(adb, "head -n 1 /proc/meminfo")
    low_power = shell_output(adb, "settings get global low_power")
    battery = shell_output(
        adb,
        "dumpsys battery | grep -E 'level:|status:|temperature:'",
    )
    return {
        "manufacturer": getprop(adb, "ro.product.manufacturer"),
        "model": getprop(adb, "ro.product.model"),
        "soc_manufacturer": getprop(adb, "ro.soc.manufacturer"),
        "soc_model": getprop(adb, "ro.soc.model"),
        "board_platform": getprop(adb, "ro.board.platform"),
        "android_version": getprop(adb, "ro.build.version.release"),
        "android_api": getprop(adb, "ro.build.version.sdk"),
        "build_fingerprint": getprop(adb, "ro.build.fingerprint"),
        "meminfo": meminfo,
        "low_power_setting": low_power,
        "battery_snapshot": battery,
        "collected_unix_time": time.time(),
    }


def battery_temperature_c(adb: str) -> float | None:
    output = shell_output(adb, "dumpsys battery | grep 'temperature:'")
    match = re.search(r"temperature:\s*(-?\d+)", output)
    return int(match.group(1)) / 10.0 if match else None


def wait_for_temperature(adb: str, maximum_c: float, poll_seconds: float) -> None:
    while True:
        temperature = battery_temperature_c(adb)
        if temperature is None:
            print("WARNING: battery temperature is unavailable; continuing.")
            return
        if temperature <= maximum_c:
            print(f"Temperature ready: {temperature:.1f} C <= {maximum_c:.1f} C")
            return
        print(
            f"Cooling: {temperature:.1f} C > {maximum_c:.1f} C; "
            f"checking again in {poll_seconds:.0f} s...",
            flush=True,
        )
        time.sleep(poll_seconds)


def install_apk(adb: str, apk: Path) -> None:
    result = run_adb(adb, ["install", "-r", "-g", str(apk)])
    print(result.stdout.strip())


def copy_to_app(adb: str, source: Path, relative_destination: str) -> None:
    remote_tmp = f"/data/local/tmp/lmbench_{source.name}"
    run_adb(adb, ["push", str(source), remote_tmp], capture=False)
    destination_parent = str(Path(relative_destination).parent)
    shell_command = (
        f"run-as {PACKAGE} mkdir -p files/{shlex.quote(destination_parent)} && "
        f"run-as {PACKAGE} cp {shlex.quote(remote_tmp)} "
        f"files/{shlex.quote(relative_destination)} && "
        f"rm -f {shlex.quote(remote_tmp)}"
    )
    run_adb(adb, ["shell", shell_command])


def app_file_exists(adb: str, relative_path: str) -> bool:
    command = f"run-as {PACKAGE} test -f files/{shlex.quote(relative_path)}"
    completed = run_adb(adb, ["shell", command], check=False)
    return completed.returncode == 0


def remove_app_file(adb: str, relative_path: str) -> None:
    command = f"run-as {PACKAGE} rm -f files/{shlex.quote(relative_path)}"
    run_adb(adb, ["shell", command], check=False)


def read_app_file(adb: str, relative_path: str) -> bytes:
    command = [adb, "exec-out", "run-as", PACKAGE, "cat", f"files/{relative_path}"]
    return subprocess.run(command, check=True, capture_output=True).stdout


def start_activity(
    adb: str,
    *,
    model_name: str,
    model_relative_path: str,
    result_name: str,
    session: int,
    warmup_runs: int,
    measured_runs: int,
    mode: str,
    image_relative_path: str | None,
    normalization: str,
    memory_interval_ms: int,
    inter_inference_delay_ms: int,
    inference_mode: str,
    cpu_threads: int,
) -> None:
    args = [
        "shell",
        "am",
        "start",
        "-W",
        "-n",
        ACTIVITY,
        "--es",
        "model_name",
        model_name,
        "--es",
        "model_path",
        model_relative_path,
        "--es",
        "result_name",
        result_name,
        "--es",
        "mode",
        mode,
        "--ei",
        "session",
        str(session),
        "--ei",
        "warmup_runs",
        str(warmup_runs),
        "--ei",
        "measured_runs",
        str(measured_runs),
        "--ei",
        "memory_interval_ms",
        str(memory_interval_ms),
        "--es",
        "normalization",
        normalization,
    ]
    args.extend(["--ei", "inter_inference_delay_ms", str(inter_inference_delay_ms),
                 "--es", "inference_mode", inference_mode,
                 "--ei", "cpu_threads", str(cpu_threads)])
    if image_relative_path:
        args.extend(["--es", "image_path", image_relative_path])
    run_adb(adb, args)


def wait_for_result(adb: str, result_path: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if app_file_exists(adb, result_path):
            return
        if app_file_exists(adb, "results/last_error.json"):
            error_data = read_app_file(adb, "results/last_error.json").decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(f"Android runner failed:\n{error_data}")
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {result_path}")


def parse_delegate(logcat: str) -> dict[str, Any]:
    matches = list(DELEGATE_RE.finditer(logcat))
    if not matches:
        return {
            "available": False,
            "total_nodes": None,
            "delegated_nodes": None,
            "gpu_coverage_percent": None,
            "cpu_fallback_nodes": None,
            "delegate_partitions": None,
        }
    delegated, total, partitions = map(int, matches[-1].groups())
    return {
        "available": True,
        "total_nodes": total,
        "delegated_nodes": delegated,
        "gpu_coverage_percent": 100.0 * delegated / total if total else None,
        "cpu_fallback_nodes": total - delegated,
        "delegate_partitions": partitions,
    }


def run_one_session(
    args: argparse.Namespace,
    model_path: Path,
    model_name: str,
    session: int,
    local_session_dir: Path,
    image_relative_path: str | None,
) -> None:
    local_session_dir.mkdir(parents=True, exist_ok=True)
    result_name = f"{model_name}_{args.mode}_session_{session:02d}.json"
    result_relative_path = f"results/{result_name}"
    remove_app_file(args.adb, result_relative_path)
    remove_app_file(args.adb, "results/last_error.json")
    run_adb(args.adb, ["shell", "am", "force-stop", PACKAGE])
    run_adb(args.adb, ["logcat", "-c"])

    before = {
        "unix_time": time.time(),
        "battery_temperature_c": battery_temperature_c(args.adb),
        "low_power_setting": shell_output(args.adb, "settings get global low_power"),
        "thermalservice": shell_output(args.adb, "dumpsys thermalservice"),
    }
    (local_session_dir / "conditions_before.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )

    start_activity(
        args.adb,
        model_name=model_name,
        model_relative_path=f"models/{model_name}.tflite",
        result_name=result_name,
        session=session,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        mode=args.mode,
        image_relative_path=image_relative_path,
        normalization=args.normalization,
        memory_interval_ms=args.memory_interval_ms,
        inter_inference_delay_ms=args.inter_inference_delay_ms,
        inference_mode=args.inference_mode,
        cpu_threads=args.cpu_threads,
    )
    wait_for_result(args.adb, result_relative_path, args.timeout_seconds)

    raw_result = read_app_file(args.adb, result_relative_path)
    result = json.loads(raw_result.decode("utf-8"))
    for key in ("inter_inference_delay_ms", "inference_mode", "cpu_threads"):
        if result.get(key) != getattr(args, key):
            raise RuntimeError(f"APK configuration mismatch for {key}: "
                               f"expected {getattr(args, key)!r}, got {result.get(key)!r}. "
                               "Rebuild and install the updated APK; do not use --skip-install.")
    logcat = run_adb(args.adb, ["logcat", "-d", "-v", "threadtime"]).stdout
    result["delegate"] = parse_delegate(logcat)
    result["host_model_path"] = str(model_path.resolve())

    after = {
        "unix_time": time.time(),
        "battery_temperature_c": battery_temperature_c(args.adb),
        "low_power_setting": shell_output(args.adb, "settings get global low_power"),
        "thermalservice": shell_output(args.adb, "dumpsys thermalservice"),
    }
    (local_session_dir / "conditions_after.json").write_text(
        json.dumps(after, indent=2), encoding="utf-8"
    )
    (local_session_dir / "logcat.txt").write_text(logcat, encoding="utf-8")
    (local_session_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Completed {model_name}, session {session}, mode={args.mode}, "
        f"p50={result['network_summary']['p50_ms']:.3f} ms, "
        f"p95={result['network_summary']['p95_ms']:.3f} ms"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--mode", choices=("latency", "memory", "pipeline"), default="latency")
    parser.add_argument("--normalization", choices=("zscore", "zero_one", "none"), default="zscore")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--measured-runs", type=int, default=100)
    parser.add_argument("--memory-interval-ms", type=int, default=50)
    parser.add_argument("--max-battery-temp-c", type=float, default=35.0)
    parser.add_argument("--temperature-poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--random-seed", type=int, default=20260912)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--inter-inference-delay-ms", type=int, default=0,
                        help="Pause after each inference except the last, including warm-up (ms)")
    parser.add_argument("--inference-mode", type=str.upper, choices=("FP16", "FP32"), default="FP16",
                        help="GPU precision policy: FP16 allows reduced precision; FP32 disallows it")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Requested thread count for CPU operators supporting this option")
    args = parser.parse_args()
    if args.inter_inference_delay_ms < 0 or args.cpu_threads < 1:
        parser.error("delay must be >= 0 and cpu-threads must be >= 1")
    if args.sessions < 1 or args.measured_runs < 1 or args.warmup_runs < 0:
        parser.error("sessions/measured-runs must be >= 1; warmup-runs must be >= 0")
    if args.temperature_poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll and timeout seconds must be > 0")
    return args


def main() -> None:
    args = parse_args()
    if not args.apk.is_file():
        raise FileNotFoundError(args.apk)
    if not args.models_dir.is_dir():
        raise NotADirectoryError(args.models_dir)
    if args.mode == "pipeline" and (args.input_image is None or not args.input_image.is_file()):
        raise ValueError("--mode pipeline requires --input-image")

    models = sorted(args.models_dir.glob("*.tflite"))
    if not models:
        raise RuntimeError(f"No .tflite files in {args.models_dir}")
    names = [safe_name(model) for model in models]
    if len(names) != len(set(names)):
        raise RuntimeError("Model filenames collide after safe-name normalization")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty: use a new directory for each experiment")
    ensure_one_device(args.adb)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_install:
        install_apk(args.adb, args.apk)

    device_info = collect_device_info(args.adb)
    device_info["protocol"] = {
        "input_shape": [1, 3, 256, 256],
        "inter_inference_delay_ms": args.inter_inference_delay_ms,
        "inference_mode": args.inference_mode,
        "cpu_threads": args.cpu_threads,
        "sessions": args.sessions,
        "warmup_runs_per_session": args.warmup_runs,
        "measured_runs_per_session": args.measured_runs,
        "mode": args.mode,
        "normalization": args.normalization if args.mode == "pipeline" else None,
        "randomization_seed": args.random_seed,
        "maximum_start_battery_temperature_c": args.max_battery_temp_c,
    }
    (args.output_dir / "device_info.json").write_text(
        json.dumps(device_info, indent=2), encoding="utf-8"
    )

    for model, model_name in zip(models, names):
        copy_to_app(args.adb, model, f"models/{model_name}.tflite")

    image_relative_path = None
    if args.input_image is not None:
        image_relative_path = f"inputs/{args.input_image.name}"
        copy_to_app(args.adb, args.input_image, image_relative_path)

    rng = random.Random(args.random_seed)
    indexed_models = list(zip(models, names))
    for session in range(1, args.sessions + 1):
        session_order = indexed_models[:]
        rng.shuffle(session_order)
        for model, model_name in session_order:
            wait_for_temperature(
                args.adb,
                args.max_battery_temp_c,
                args.temperature_poll_seconds,
            )
            local_session_dir = (
                args.output_dir / args.mode / model_name / f"session_{session:02d}"
            )
            run_one_session(
                args,
                model,
                model_name,
                session,
                local_session_dir,
                image_relative_path,
            )

    print("All sessions completed.")
    print(
        "Next: python tools/summarize_mobile_results.py "
        f"--results-dir {shlex.quote(str(args.output_dir))}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
