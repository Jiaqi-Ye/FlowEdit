import argparse
import csv
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)


def build_prompt_lookup(dataset_yaml):
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        dataset = yaml.load(f, Loader=yaml.FullLoader)

    lookup = {}
    for item in dataset:
        src = item["input_img"]
        for i, prompt in enumerate(item["target_prompts"]):
            lookup[(src, str(i))] = prompt
    return lookup


def find_output_png(output_dir):
    matches = glob.glob(os.path.join(output_dir, "output_solver_*.png"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


@torch.no_grad()
def clip_alignment_score(model, processor, image, prompt, device):
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

    # Newer Transformers versions can return model-output objects from the
    # get_*_features helpers. Calling the full CLIP model gives projected
    # image/text embeddings with matching dimensions across versions.
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return torch.sum(image_features * text_features, dim=-1).item()


@torch.no_grad()
def dino_similarity_score(model, processor, source_image, edited_image, device):
    src_inputs = processor(images=source_image, return_tensors="pt").to(device)
    edt_inputs = processor(images=edited_image, return_tensors="pt").to(device)

    src_feat = model(**src_inputs).last_hidden_state[:, 0, :]
    edt_feat = model(**edt_inputs).last_hidden_state[:, 0, :]

    src_feat = F.normalize(src_feat, dim=-1)
    edt_feat = F.normalize(edt_feat, dim=-1)
    return torch.sum(src_feat * edt_feat, dim=-1).item()


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_summary_csv", default="outputs/run_summary.csv")
    parser.add_argument("--dataset_yaml", default="edits_midpoint_eval.yaml")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino_model", default="facebook/dinov2-base")
    parser.add_argument("--out_samples", default="outputs/metrics/clip_dino_per_sample.csv")
    parser.add_argument("--out_summary", default="outputs/metrics/clip_dino_summary.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    prompt_lookup = build_prompt_lookup(args.dataset_yaml)

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
    dino_model = AutoModel.from_pretrained(args.dino_model).to(device).eval()

    per_sample_rows = []

    with open(args.run_summary_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_image_path = row["source_image"]
            target_index = row["target_index"]
            output_dir = row["output_dir"]

            target_prompt = prompt_lookup.get((source_image_path, target_index))
            if target_prompt is None:
                print(f"Skip (prompt not found): {source_image_path}, target_index={target_index}")
                continue

            output_image_path = find_output_png(output_dir)
            if output_image_path is None:
                print(f"Skip (output not found): {output_dir}")
                continue

            source_image = Image.open(source_image_path).convert("RGB")
            edited_image = Image.open(output_image_path).convert("RGB")

            clip_score = clip_alignment_score(
                clip_model, clip_processor, edited_image, target_prompt, device
            )
            dino_score = dino_similarity_score(
                dino_model, dino_processor, source_image, edited_image, device
            )

            row_out = {
                "exp_name": row["exp_name"],
                "solver_type": row["solver_type"],
                "estimated_nfe": row.get("estimated_nfe", ""),
                "pc_guidance_lambda": row.get("pc_guidance_lambda", ""),
                "pc_guidance_gamma": row.get("pc_guidance_gamma", ""),
                "source_image": source_image_path,
                "target_index": target_index,
                "elapsed_seconds": row["elapsed_seconds"],
                "output_image": output_image_path,
                "clip_alignment": f"{clip_score:.6f}",
                "dino_similarity": f"{dino_score:.6f}",
                "edit_preservation_score": f"{(clip_score * dino_score):.6f}",
            }
            per_sample_rows.append(row_out)

    if not per_sample_rows:
        raise RuntimeError("No valid samples were evaluated. Check run_summary.csv and dataset_yaml.")

    out_samples = Path(args.out_samples)
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    with open(out_samples, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"Wrote per-sample metrics: {out_samples}")

    grouped = {}
    for row in per_sample_rows:
        key = (
            row["exp_name"],
            row["solver_type"],
            row["estimated_nfe"],
            row["pc_guidance_lambda"],
            row["pc_guidance_gamma"],
        )
        grouped.setdefault(key, {"clip": [], "dino": [], "score": [], "time": []})
        grouped[key]["clip"].append(float(row["clip_alignment"]))
        grouped[key]["dino"].append(float(row["dino_similarity"]))
        grouped[key]["score"].append(float(row["edit_preservation_score"]))
        grouped[key]["time"].append(float(row["elapsed_seconds"]))

    summary_rows = []
    for (exp_name, solver_type, estimated_nfe, pc_guidance_lambda, pc_guidance_gamma), vals in grouped.items():
        summary_rows.append(
            {
                "exp_name": exp_name,
                "solver_type": solver_type,
                "estimated_nfe": estimated_nfe,
                "pc_guidance_lambda": pc_guidance_lambda,
                "pc_guidance_gamma": pc_guidance_gamma,
                "num_samples": len(vals["clip"]),
                "clip_alignment_mean": f"{mean(vals['clip']):.6f}",
                "dino_similarity_mean": f"{mean(vals['dino']):.6f}",
                "edit_preservation_score_mean": f"{mean(vals['score']):.6f}",
                "elapsed_seconds_mean": f"{mean(vals['time']):.3f}",
            }
        )

    if not summary_rows:
        raise RuntimeError("No summary rows generated.")

    out_summary = Path(args.out_summary)
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary metrics: {out_summary}")


if __name__ == "__main__":
    main()
