import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def latest_output(exp_name, model_type, source_name, target_index):
    pattern = (
        f"outputs/{exp_name}/{model_type}/src_{source_name}/tar_{target_index}/"
        "output_solver_*.png"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def runtime_lookup(summary_path):
    if not os.path.exists(summary_path):
        return {}

    rows = {}
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["exp_name"],
                row["model_type"],
                Path(row["source_image"]).stem,
                row["target_index"],
            )
            rows[key] = row.get("elapsed_seconds", "")
    return rows


def fit_image(image, size):
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def mean_abs_diff(image_a, image_b):
    arr_a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(image_b.convert("RGB").resize(image_a.size), dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)) / 255.0)


def draw_label(draw, xy, text, font, width):
    x, y = xy
    draw.rectangle((x, y, x + width, y + 34), fill="white")
    draw.text((x + 8, y + 8), text, fill="black", font=font)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_yaml", default="edits.yaml")
    parser.add_argument("--model_type", default="SD3")
    parser.add_argument("--euler_exp", default="FlowEdit_SD3_Euler")
    parser.add_argument("--midpoint_exp", default="FlowEdit_SD3_Midpoint")
    parser.add_argument("--summary_csv", default="outputs/run_summary.csv")
    parser.add_argument("--out", default="outputs/solver_comparison/sd3_euler_vs_midpoint.png")
    args = parser.parse_args()

    with open(args.dataset_yaml) as f:
        dataset = yaml.load(f, Loader=yaml.FullLoader)

    runtimes = runtime_lookup(args.summary_csv)
    tile_size = (320, 320)
    label_height = 52
    columns = ["Input", "Euler", "Midpoint"]
    rows = []

    for data in dataset:
        source_path = data["input_img"]
        source_name = Path(source_path).stem
        for target_index, target_prompt in enumerate(data["target_prompts"]):
            euler_path = latest_output(args.euler_exp, args.model_type, source_name, target_index)
            midpoint_path = latest_output(args.midpoint_exp, args.model_type, source_name, target_index)
            if euler_path is None or midpoint_path is None:
                print(f"Skipping {source_name}/tar_{target_index}: missing solver output")
                continue
            rows.append((source_path, source_name, target_index, target_prompt, euler_path, midpoint_path))

    if not rows:
        raise RuntimeError("No matched Euler/Midpoint output pairs found.")

    width = len(columns) * tile_size[0]
    height = len(rows) * (tile_size[1] + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_index, (source_path, source_name, target_index, _, euler_path, midpoint_path) in enumerate(rows):
        y0 = row_index * (tile_size[1] + label_height)
        input_img = fit_image(Image.open(source_path), tile_size)
        euler_img = fit_image(Image.open(euler_path), tile_size)
        midpoint_img = fit_image(Image.open(midpoint_path), tile_size)

        sheet.paste(input_img, (0, y0 + label_height))
        sheet.paste(euler_img, (tile_size[0], y0 + label_height))
        sheet.paste(midpoint_img, (2 * tile_size[0], y0 + label_height))

        diff = mean_abs_diff(Image.open(euler_path), Image.open(midpoint_path))
        euler_time = runtimes.get((args.euler_exp, args.model_type, source_name, str(target_index)), "")
        midpoint_time = runtimes.get((args.midpoint_exp, args.model_type, source_name, str(target_index)), "")

        labels = [
            f"Input: {source_name}",
            f"Euler | {euler_time}s",
            f"Midpoint | {midpoint_time}s | MAD {diff:.4f}",
        ]
        for col, label in enumerate(labels):
            draw_label(draw, (col * tile_size[0], y0), label, font, tile_size[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Wrote comparison sheet to {out_path}")


if __name__ == "__main__":
    main()
