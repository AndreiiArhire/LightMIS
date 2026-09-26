#!/usr/bin/env python3
"""Aggregate Android session JSON files into run-level and model-level CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty list")
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("**/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("error") not in (None, ""):
            raise RuntimeError(f"Failed result in {path}: {data['error']}")
        data["_path"] = str(path)
        results.append(data)
    if not results:
        raise RuntimeError(f"No result.json files under {root}")
    return results


def write_runs_csv(results: list[dict[str, Any]], destination: Path) -> None:
    fields = ["model", "mode", "inference_mode", "cpu_threads", "inter_inference_delay_ms", "session", "run", "network_latency_ms", "pipeline_latency_ms"]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            network = result.get("network_latency_ms", [])
            pipeline = result.get("pipeline_latency_ms", [])
            for index, network_ms in enumerate(network, start=1):
                writer.writerow(
                    {
                        "model": result["model"],
                        "mode": result["mode"],
                        "inference_mode": result.get("inference_mode", "unknown"),
                        "cpu_threads": result.get("cpu_threads", "unknown"),
                        "inter_inference_delay_ms": result.get("inter_inference_delay_ms", "unknown"),
                        "session": result["session"],
                        "run": index,
                        "network_latency_ms": f"{float(network_ms):.9f}",
                        "pipeline_latency_ms": (
                            f"{float(pipeline[index - 1]):.9f}"
                            if index <= len(pipeline)
                            else ""
                        ),
                    }
                )


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "sample_sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "p50_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "max_ms": max(values),
    }


def write_summary_csv(results: list[dict[str, Any]], destination: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["model"], result["mode"])].append(result)

    # Refuse to pool different settings or legacy unknown settings for one model/mode.
    config_keys = ("inference_mode", "cpu_threads", "inter_inference_delay_ms",
                   "litert_version", "gpu_precision_loss_allowed", "warmup_runs",
                   "measured_runs", "normalization")
    for group, rows in grouped.items():
        configurations = {tuple(row.get(k) for k in config_keys) for row in rows}
        if len(configurations) > 1:
            raise ValueError(f"Mixed configurations for {group}; summarize separate experiment folders")

    fields = [
        "model", "mode", "inference_mode", "cpu_threads", "inter_inference_delay_ms", "sessions", "warmup_per_session", "runs_per_session",
        "measured_total", "network_mean_ms", "network_sd_ms", "network_p50_ms",
        "network_p95_ms", "pipeline_p50_ms", "pipeline_p95_ms", "initialization_mean_ms",
        "model_file_mb_decimal", "peak_process_rss_mb", "tflite_nodes",
        "delegated_nodes", "gpu_coverage_percent", "cpu_fallback_nodes",
        "delegate_partitions", "temperature_min_c", "temperature_max_c",
        "power_save_seen", "litert_version", "input_shape", "input_dtype", "output_shape",
        "output_dtype",
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (model, mode), rows in sorted(grouped.items()):
            network = [float(v) for row in rows for v in row["network_latency_ms"]]
            pipeline = [float(v) for row in rows for v in row.get("pipeline_latency_ms", [])]
            net = summarize(network)
            pipe = summarize(pipeline) if pipeline else None
            delegate_rows = [row.get("delegate", {}) for row in rows]
            delegate = next((item for item in delegate_rows if item.get("available")), {})
            temperatures = [
                float(value)
                for row in rows
                for value in (
                    row.get("battery_temperature_start_c"),
                    row.get("battery_temperature_end_c"),
                )
                if value is not None
            ]
            peaks = [
                float(row["peak_process_rss_kb"]) / 1024.0
                for row in rows
                if row.get("peak_process_rss_kb") not in (None, 0)
            ]
            writer.writerow(
                {
                    "model": model,
                    "mode": mode,
                    "inference_mode": rows[0].get("inference_mode", "unknown"),
                    "cpu_threads": rows[0].get("cpu_threads", "unknown"),
                    "inter_inference_delay_ms": rows[0].get("inter_inference_delay_ms", "unknown"),
                    "sessions": len(rows),
                    "warmup_per_session": rows[0]["warmup_runs"],
                    "runs_per_session": rows[0]["measured_runs"],
                    "measured_total": net["n"],
                    "network_mean_ms": f"{net['mean_ms']:.6f}",
                    "network_sd_ms": f"{net['sample_sd_ms']:.6f}",
                    "network_p50_ms": f"{net['p50_ms']:.6f}",
                    "network_p95_ms": f"{net['p95_ms']:.6f}",
                    "pipeline_p50_ms": f"{pipe['p50_ms']:.6f}" if pipe else "",
                    "pipeline_p95_ms": f"{pipe['p95_ms']:.6f}" if pipe else "",
                    "initialization_mean_ms": f"{statistics.fmean(float(row['initialization_ms']) for row in rows):.6f}",
                    "model_file_mb_decimal": f"{float(rows[0]['model_file_bytes']) / 1_000_000:.6f}",
                    "peak_process_rss_mb": f"{max(peaks):.6f}" if peaks else "",
                    "tflite_nodes": delegate.get("total_nodes", ""),
                    "delegated_nodes": delegate.get("delegated_nodes", ""),
                    "gpu_coverage_percent": delegate.get("gpu_coverage_percent", ""),
                    "cpu_fallback_nodes": delegate.get("cpu_fallback_nodes", ""),
                    "delegate_partitions": delegate.get("delegate_partitions", ""),
                    "temperature_min_c": f"{min(temperatures):.1f}" if temperatures else "",
                    "temperature_max_c": f"{max(temperatures):.1f}" if temperatures else "",
                    "power_save_seen": any(
                        bool(row.get("power_save_start")) or bool(row.get("power_save_end"))
                        for row in rows
                    ),
                    "litert_version": rows[0].get("litert_version", ""),
                    "input_shape": "x".join(map(str, rows[0].get("input_shape", []))),
                    "input_dtype": rows[0].get("input_dtype", ""),
                    "output_shape": "x".join(map(str, rows[0].get("output_shape", []))),
                    "output_dtype": rows[0].get("output_dtype", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results = load_results(args.results_dir)
    write_runs_csv(results, args.results_dir / "mobile_runs.csv")
    write_summary_csv(results, args.results_dir / "mobile_summary.csv")
    print(args.results_dir / "mobile_runs.csv")
    print(args.results_dir / "mobile_summary.csv")


if __name__ == "__main__":
    main()
