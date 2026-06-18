import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image


def find_output_png(output_dir):
    matches = glob.glob(os.path.join(output_dir, "output_solver_*.png"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def image_artifact_proxy(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    channel_max = arr.max(axis=-1)
    channel_min = arr.min(axis=-1)
    saturation = (channel_max - channel_min) / np.maximum(channel_max, 1e-6)

    clipping_ratio = np.mean((arr <= 0.01) | (arr >= 0.99))
    high_saturation_ratio = np.mean((saturation >= 0.95) & (channel_max >= 0.2))
    artifact_proxy = clipping_ratio + high_saturation_ratio

    return {
        "clipping_ratio": clipping_ratio,
        "high_saturation_ratio": high_saturation_ratio,
        "artifact_proxy": artifact_proxy,
    }


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_summary_csv", default="outputs/run_summary.csv")
    parser.add_argument("--out_samples", default="outputs/metrics/artifact_proxy_per_sample.csv")
    parser.add_argument("--out_summary", default="outputs/metrics/artifact_proxy_summary.csv")
    args = parser.parse_args()

    per_sample_rows = []

    with open(args.run_summary_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            output_image_path = find_output_png(row["output_dir"])
            if output_image_path is None:
                print(f"Skip (output not found): {row['output_dir']}")
                continue

            source_image = Image.open(row["source_image"]).convert("RGB")
            edited_image = Image.open(output_image_path).convert("RGB")

            source_metrics = image_artifact_proxy(source_image)
            edited_metrics = image_artifact_proxy(edited_image)

            row_out = {
                "exp_name": row["exp_name"],
                "solver_type": row["solver_type"],
                "normalized_solver_type": row.get("normalized_solver_type", ""),
                "theory_family": row.get("theory_family", ""),
                "theory_formula": row.get("theory_formula", ""),
                "midpoint_space": row.get("midpoint_space", ""),
                "correction_mode": row.get("correction_mode", ""),
                "estimated_nfe": row.get("estimated_nfe", ""),
                "actual_nfe": row.get("actual_nfe", ""),
                "pc_guidance_lambda": row.get("pc_guidance_lambda", ""),
                "pc_guidance_gamma": row.get("pc_guidance_gamma", ""),
                "pc_guidance_weight": row.get("pc_guidance_weight", ""),
                "pc_magnitude_weight": row.get("pc_magnitude_weight", ""),
                "pc_magnitude_min": row.get("pc_magnitude_min", ""),
                "pc_magnitude_max": row.get("pc_magnitude_max", ""),
                "source_image": row["source_image"],
                "target_index": row["target_index"],
                "output_image": output_image_path,
                "source_artifact_proxy": f"{source_metrics['artifact_proxy']:.6f}",
                "edited_artifact_proxy": f"{edited_metrics['artifact_proxy']:.6f}",
                "artifact_proxy_delta_vs_source": f"{(edited_metrics['artifact_proxy'] - source_metrics['artifact_proxy']):.6f}",
                "edited_clipping_ratio": f"{edited_metrics['clipping_ratio']:.6f}",
                "edited_high_saturation_ratio": f"{edited_metrics['high_saturation_ratio']:.6f}",
            }
            per_sample_rows.append(row_out)

    if not per_sample_rows:
        raise RuntimeError("No valid samples were evaluated. Check run_summary.csv outputs.")

    out_samples = Path(args.out_samples)
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    with open(out_samples, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"Wrote per-sample artifact proxy metrics: {out_samples}")

    grouped = {}
    for row in per_sample_rows:
        key = (
            row["exp_name"],
            row["solver_type"],
            row["normalized_solver_type"],
            row["theory_family"],
            row["theory_formula"],
            row["midpoint_space"],
            row["correction_mode"],
            row["estimated_nfe"],
            row["actual_nfe"],
            row["pc_guidance_lambda"],
            row["pc_guidance_gamma"],
            row["pc_guidance_weight"],
            row.get("pc_magnitude_weight", ""),
            row.get("pc_magnitude_min", ""),
            row.get("pc_magnitude_max", ""),
        )
        grouped.setdefault(key, {"artifact": [], "delta": [], "clip": [], "sat": []})
        grouped[key]["artifact"].append(float(row["edited_artifact_proxy"]))
        grouped[key]["delta"].append(float(row["artifact_proxy_delta_vs_source"]))
        grouped[key]["clip"].append(float(row["edited_clipping_ratio"]))
        grouped[key]["sat"].append(float(row["edited_high_saturation_ratio"]))

    summary_rows = []
    for (
        exp_name,
        solver_type,
        normalized_solver_type,
        theory_family,
        theory_formula,
        midpoint_space,
        correction_mode,
        estimated_nfe,
        actual_nfe,
        pc_guidance_lambda,
        pc_guidance_gamma,
        pc_guidance_weight,
        pc_magnitude_weight,
        pc_magnitude_min,
        pc_magnitude_max,
    ), vals in grouped.items():
        summary_rows.append(
            {
                "exp_name": exp_name,
                "solver_type": solver_type,
                "normalized_solver_type": normalized_solver_type,
                "theory_family": theory_family,
                "theory_formula": theory_formula,
                "midpoint_space": midpoint_space,
                "correction_mode": correction_mode,
                "estimated_nfe": estimated_nfe,
                "actual_nfe": actual_nfe,
                "pc_guidance_lambda": pc_guidance_lambda,
                "pc_guidance_gamma": pc_guidance_gamma,
                "pc_guidance_weight": pc_guidance_weight,
                "pc_magnitude_weight": pc_magnitude_weight,
                "pc_magnitude_min": pc_magnitude_min,
                "pc_magnitude_max": pc_magnitude_max,
                "num_samples": len(vals["artifact"]),
                "edited_artifact_proxy_mean": f"{mean(vals['artifact']):.6f}",
                "artifact_proxy_delta_vs_source_mean": f"{mean(vals['delta']):.6f}",
                "edited_clipping_ratio_mean": f"{mean(vals['clip']):.6f}",
                "edited_high_saturation_ratio_mean": f"{mean(vals['sat']):.6f}",
            }
        )

    out_summary = Path(args.out_summary)
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote artifact proxy summary: {out_summary}")


if __name__ == "__main__":
    main()
