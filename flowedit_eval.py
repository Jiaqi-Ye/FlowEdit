"""FlowEdit released-dataset evaluation helpers.

This module keeps the full evaluation protocol usable from both Colab and
command-line runs. Heavy metric dependencies are imported only when metrics are
actually computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


METRIC_DIRECTIONS = {
    "CLIP-T": "higher",
    "CLIP-I": "higher",
    "LPIPS": "lower",
    "DINO": "higher",
    "DreamSim": "lower",
}

DEFAULT_METHODS = ("flowedit_baseline", "bridge_interpolate", "bridge_directional")
METHOD_SOLVERS = {
    "flowedit_baseline": "euler",
    "original_flowedit": "euler",
    "baseline": "euler",
    "bridge_interpolate": "flowedit_bridge_interpolate",
    "bridge_directional": "flowedit_bridge_directional",
}

def format_setting_float(value: Any) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "")


def default_bridge_setting_id(setting: Dict[str, Any]) -> str:
    lam = format_setting_float(setting.get("pc_guidance_lambda", 1.0))
    gamma = format_setting_float(setting.get("pc_guidance_gamma", 1.0))
    weight = format_setting_float(setting.get("pc_guidance_weight", 1.0))
    enable = setting.get("pc_enable_below_t", 1.0)
    setting_id = f"l{lam}_g{gamma}_w{weight}"
    if float(enable) < 1.0:
        setting_id += f"_e{format_setting_float(enable)}"
    return setting_id


def parse_bridge_settings(value: str | None) -> Optional[List[Dict[str, Any]]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--bridge_settings_json must be valid JSON.") from exc
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("--bridge_settings_json must be a JSON object or a JSON list of objects.")

    settings: List[Dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Bridge setting {index} must be a JSON object.")
        setting = dict(item)
        setting.setdefault("pc_guidance_lambda", 1.0)
        setting.setdefault("pc_guidance_gamma", 1.0)
        setting.setdefault("pc_guidance_weight", 1.0)
        setting.setdefault("pc_enable_below_t", 1.0)
        setting.setdefault("setting_id", default_bridge_setting_id(setting))
        settings.append(setting)
    return settings


MODEL_DEFAULTS = {
    "sd3": {
        "model_type": "SD3",
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "T_steps": 50,
        "n_avg": 1,
        "src_guidance_scale": 3.5,
        "tar_guidance_scale": 13.5,
        "n_min": 0,
        "n_max": 33,
        "seed": 42,
    },
    "flux": {
        "model_type": "FLUX",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "T_steps": 28,
        "n_avg": 1,
        "src_guidance_scale": 1.5,
        "tar_guidance_scale": 5.5,
        "n_min": 0,
        "n_max": 24,
        "seed": 10,
    },
}

IMAGE_NAME_ALIASES = {
    "coconut_out.png": "coconut.png",
    "meditation1.png": "meditation2.png",
}


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    image_id: str
    target_index: int
    target_code: str
    source_image_path: str
    source_prompt: str
    target_prompt: str
    negative_prompt: str = ""


def slugify(value: Any, fallback: str = "item") -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def model_slug(model_name_or_type: str) -> str:
    value = str(model_name_or_type).strip().lower()
    if value in {"stable_diffusion_3", "stable-diffusion-3", "sd3"}:
        return "sd3"
    if value in {"flux", "flux.1", "flux1"}:
        return "flux"
    return slugify(value, fallback="model")


def _repo_relative(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    repo_root = repo_root.resolve()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def resolve_source_image_path(
    raw_path: str,
    dataset_yaml: Path,
    repo_root: Path,
    data_images_dir: str = "Data/Images",
) -> str:
    raw_path = str(raw_path).strip()
    if not raw_path:
        raise ValueError("Dataset item is missing input_img/init_img.")

    raw = Path(raw_path)
    image_name = raw.name
    alias_name = IMAGE_NAME_ALIASES.get(image_name, image_name)
    candidates = [
        repo_root / raw_path,
        dataset_yaml.parent / raw_path,
        repo_root / data_images_dir / image_name,
        repo_root / data_images_dir / alias_name,
        dataset_yaml.parent / "Images" / image_name,
        dataset_yaml.parent / "Images" / alias_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return _repo_relative(candidate, repo_root)

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve source image '{raw_path}'. Searched:\n{searched}")


def load_eval_samples(
    dataset_yaml: str | Path,
    sample_limit: Optional[int] = None,
    sample_offset: int = 0,
    repo_root: str | Path = ".",
    data_images_dir: str = "Data/Images",
) -> List[EvalSample]:
    repo_root = Path(repo_root)
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.is_absolute():
        dataset_yaml = repo_root / dataset_yaml

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        dataset = yaml.safe_load(f) or []

    samples: List[EvalSample] = []
    seen_ids: Dict[str, int] = {}
    for item in dataset:
        raw_image = item.get("input_img") or item.get("init_img")
        source_image_path = resolve_source_image_path(
            raw_image,
            dataset_yaml=dataset_yaml,
            repo_root=repo_root,
            data_images_dir=data_images_dir,
        )
        source_prompt = str(item.get("source_prompt", "")).strip()
        negative_prompt = str(item.get("negative_prompt", "") or "").strip()
        target_prompts = item.get("target_prompts") or []
        target_codes = item.get("target_codes") or []
        image_id = slugify(Path(source_image_path).stem, fallback="image")

        for target_index, target_prompt in enumerate(target_prompts):
            target_code = (
                str(target_codes[target_index]).strip()
                if target_index < len(target_codes)
                else f"target_{target_index:03d}"
            )
            base_sample_id = f"{image_id}__{slugify(target_code, fallback=f'target_{target_index:03d}')}"
            duplicate_count = seen_ids.get(base_sample_id, 0)
            seen_ids[base_sample_id] = duplicate_count + 1
            sample_id = base_sample_id if duplicate_count == 0 else f"{base_sample_id}_{duplicate_count + 1}"
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    image_id=image_id,
                    target_index=target_index,
                    target_code=target_code,
                    source_image_path=source_image_path,
                    source_prompt=source_prompt,
                    target_prompt=str(target_prompt).strip(),
                    negative_prompt=negative_prompt,
                )
            )

    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative.")
    if sample_limit is not None and sample_limit < 0:
        raise ValueError("sample_limit must be non-negative or omitted for full dataset.")

    selected = samples[sample_offset:]
    if sample_limit is not None:
        selected = selected[:sample_limit]
    return selected


def method_name_from_solver(solver_type: str) -> str:
    solver_type = str(solver_type).lower()
    if solver_type == "euler":
        return "flowedit_baseline"
    if solver_type in {"flowedit_bridge_interpolate", "flowedit_bridge_interp", "flowedit_bridge_interpolation"}:
        return "bridge_interpolate"
    if solver_type in {"flowedit_bridge_directional", "flowedit_bridge_dir", "flowedit_bridge_direction"}:
        return "bridge_directional"
    return slugify(solver_type, fallback="method")


def build_default_experiments(
    model_name: str,
    dataset_yaml: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    budget_mode: str = "paper",
    src_guidance_scale: Optional[float] = None,
    tar_guidance_scale: Optional[float] = None,
    bridge_settings: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    model_key = model_slug(model_name)
    if model_key not in MODEL_DEFAULTS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Use 'sd3' or 'flux'.")
    if budget_mode not in {"paper", "same_nfe"}:
        raise ValueError("budget_mode must be 'paper' or 'same_nfe'.")

    base_defaults = MODEL_DEFAULTS[model_key]
    experiments: List[Dict[str, Any]] = []
    for method in methods:
        method_key = slugify(method)
        if method_key not in METHOD_SOLVERS:
            raise ValueError(f"Unsupported method '{method}'. Available: {sorted(METHOD_SOLVERS)}")
        solver_type = METHOD_SOLVERS[method_key]
        cfg = dict(base_defaults)
        if src_guidance_scale is not None:
            cfg["src_guidance_scale"] = src_guidance_scale
        if tar_guidance_scale is not None:
            cfg["tar_guidance_scale"] = tar_guidance_scale
        cfg.update(
            {
                "exp_name": f"{model_key}_{method_key}",
                "dataset_yaml": dataset_yaml,
                "sampler_type": f"FlowEdit{base_defaults['model_type']}",
                "solver_type": solver_type,
                "method_name": method_key,
                "method_family": method_key,
                "setting_id": "default",
            }
        )
        if budget_mode == "same_nfe" and method_key in {"bridge_interpolate", "bridge_directional"}:
            cfg["n_max"] = max(1, math.ceil(base_defaults["n_max"] / 2))
            cfg["T_steps"] = max(cfg["n_max"], math.ceil(base_defaults["T_steps"] / 2))
        if bridge_settings and method_key in {"bridge_interpolate", "bridge_directional"}:
            for setting in bridge_settings:
                setting_cfg = dict(cfg)
                setting_id = str(setting.get("setting_id") or default_bridge_setting_id(setting))
                setting_cfg.update(
                    {
                        "exp_name": f"{model_key}_{method_key}_{slugify(setting_id)}",
                        "setting_id": setting_id,
                        "pc_guidance_lambda": float(setting.get("pc_guidance_lambda", 1.0)),
                        "pc_guidance_gamma": float(setting.get("pc_guidance_gamma", 1.0)),
                        "pc_guidance_weight": float(setting.get("pc_guidance_weight", 1.0)),
                        "pc_enable_below_t": float(setting.get("pc_enable_below_t", 1.0)),
                    }
                )
                for key in ("src_guidance_scale", "tar_guidance_scale", "T_steps", "n_min", "n_max", "n_avg", "seed"):
                    if key in setting:
                        setting_cfg[key] = setting[key]
                experiments.append(setting_cfg)
        else:
            experiments.append(cfg)
    return experiments


def write_experiment_yaml(
    output_yaml: str | Path,
    model_name: str,
    dataset_yaml: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    budget_mode: str = "paper",
    src_guidance_scale: Optional[float] = None,
    tar_guidance_scale: Optional[float] = None,
    bridge_settings: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    experiments = build_default_experiments(
        model_name=model_name,
        dataset_yaml=dataset_yaml,
        methods=methods,
        budget_mode=budget_mode,
        src_guidance_scale=src_guidance_scale,
        tar_guidance_scale=tar_guidance_scale,
        bridge_settings=bridge_settings,
    )
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(experiments, f, sort_keys=False, allow_unicode=False)
    return experiments


def build_eval_output_paths(
    output_root: str | Path,
    model_name: str,
    method_name: str,
    sample_id: str,
    setting_id: str = "default",
) -> Dict[str, Path]:
    method_root = Path(output_root) / model_slug(model_name) / slugify(method_name, fallback="method")
    if setting_id and setting_id not in {"default", method_name}:
        method_root = method_root / slugify(setting_id, fallback="setting")
    image_dir = method_root / "images"
    metadata_dir = method_root / "metadata"
    return {
        "method_root": method_root,
        "image_dir": image_dir,
        "metadata_dir": metadata_dir,
        "output_image": image_dir / f"{slugify(sample_id, fallback='sample')}.png",
        "metadata_json": metadata_dir / f"{slugify(sample_id, fallback='sample')}.json",
        "prompts_txt": metadata_dir / f"{slugify(sample_id, fallback='sample')}_prompts.txt",
    }


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def resize_longest_side(image, image_resolution: Optional[int]):
    if image_resolution is None or image_resolution <= 0:
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side == image_resolution:
        return image
    scale = image_resolution / longest_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resample = getattr(getattr(image, "Resampling", image), "BICUBIC", 3)
    return image.resize((new_width, new_height), resample=resample)


def parse_metric_names(metrics: str | Sequence[str] | None) -> List[str]:
    if metrics is None:
        return list(METRIC_DIRECTIONS)
    if isinstance(metrics, str):
        requested = [part.strip() for part in metrics.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in metrics if str(part).strip()]
    aliases = {key.lower(): key for key in METRIC_DIRECTIONS}
    selected = []
    for metric in requested:
        canonical = aliases.get(metric.lower())
        if canonical is None:
            raise ValueError(f"Unknown metric '{metric}'. Available: {', '.join(METRIC_DIRECTIONS)}")
        if canonical not in selected:
            selected.append(canonical)
    return selected


class MetricComputer:
    def __init__(
        self,
        device: Optional[str] = None,
        clip_model: str = "openai/clip-vit-base-patch32",
        dino_model: str = "facebook/dino-vitb16",
        lpips_net: str = "alex",
        lpips_resize: int = 0,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model
        self.dino_model_name = dino_model
        self.lpips_net = lpips_net
        self.lpips_resize = lpips_resize
        self._clip_model = None
        self._clip_processor = None
        self._dino_model = None
        self._dino_processor = None
        self._lpips_model = None
        self._dreamsim_model = None
        self._dreamsim_preprocess = None

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        return self._clip_model, self._clip_processor

    def _load_dino(self):
        if self._dino_model is None:
            from transformers import AutoImageProcessor, AutoModel

            self._dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_name)
            self._dino_model = AutoModel.from_pretrained(self.dino_model_name).to(self.device).eval()
        return self._dino_model, self._dino_processor

    def _load_lpips(self):
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("Install lpips to compute LPIPS: pip install lpips") from exc

            self._lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device).eval()
        return self._lpips_model

    def _load_dreamsim(self):
        if self._dreamsim_model is None:
            try:
                from dreamsim import dreamsim
            except ImportError as exc:
                raise ImportError("Install DreamSim to compute DreamSim: pip install dreamsim") from exc

            self._dreamsim_model, self._dreamsim_preprocess = dreamsim(pretrained=True, device=self.device)
            self._dreamsim_model = self._dreamsim_model.eval()
        return self._dreamsim_model, self._dreamsim_preprocess

    def clip_t(self, edited_image, target_prompt: str) -> float:
        import torch.nn.functional as F

        model, processor = self._load_clip()
        inputs = processor(text=[target_prompt], images=edited_image, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            outputs = model(**inputs)
            image_features = F.normalize(outputs.image_embeds, dim=-1)
            text_features = F.normalize(outputs.text_embeds, dim=-1)
        return float((image_features * text_features).sum(dim=-1).item())

    def clip_i(self, source_image, edited_image) -> float:
        import torch.nn.functional as F

        model, processor = self._load_clip()
        inputs = processor(images=[source_image, edited_image], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            image_features = model.get_image_features(**inputs)
            if not self.torch.is_tensor(image_features):
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                    if hasattr(model, "visual_projection"):
                        image_features = model.visual_projection(image_features)
                elif hasattr(image_features, "last_hidden_state"):
                    image_features = image_features.last_hidden_state[:, 0, :]
                elif isinstance(image_features, (tuple, list)) and image_features:
                    image_features = image_features[0]
                else:
                    raise TypeError(f"Unsupported CLIP image feature output: {type(image_features)!r}")
            image_features = F.normalize(image_features, dim=-1)
        return float((image_features[0] * image_features[1]).sum().item())

    def dino(self, source_image, edited_image) -> float:
        import torch.nn.functional as F

        model, processor = self._load_dino()
        src_inputs = processor(images=source_image, return_tensors="pt").to(self.device)
        edt_inputs = processor(images=edited_image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            src_out = model(**src_inputs)
            edt_out = model(**edt_inputs)
            src_feat = getattr(src_out, "pooler_output", None)
            edt_feat = getattr(edt_out, "pooler_output", None)
            if src_feat is None:
                src_feat = src_out.last_hidden_state[:, 0, :]
            if edt_feat is None:
                edt_feat = edt_out.last_hidden_state[:, 0, :]
            src_feat = F.normalize(src_feat, dim=-1)
            edt_feat = F.normalize(edt_feat, dim=-1)
        return float((src_feat * edt_feat).sum(dim=-1).item())

    def _lpips_tensor(self, image):
        import numpy as np

        if self.lpips_resize and self.lpips_resize > 0:
            image = image.resize((self.lpips_resize, self.lpips_resize))
        arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        tensor = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (tensor.to(self.device) * 2.0) - 1.0

    def lpips(self, source_image, edited_image) -> float:
        model = self._load_lpips()
        if source_image.size != edited_image.size:
            edited_image = edited_image.resize(source_image.size)
        src = self._lpips_tensor(source_image)
        edt = self._lpips_tensor(edited_image)
        with self.torch.inference_mode():
            return float(model(src, edt).item())

    def dreamsim(self, source_image, edited_image) -> float:
        model, preprocess = self._load_dreamsim()
        src = preprocess(source_image).to(self.device)
        edt = preprocess(edited_image).to(self.device)
        if src.ndim == 3:
            src = src.unsqueeze(0)
        if edt.ndim == 3:
            edt = edt.unsqueeze(0)
        with self.torch.inference_mode():
            return float(model(src, edt).item())

    def compute_selected(
        self,
        source_image,
        edited_image,
        target_prompt: str,
        metrics: Sequence[str],
        skip_failed_metrics: bool = False,
    ) -> Dict[str, Optional[float]]:
        metric_fns = {
            "CLIP-T": lambda: self.clip_t(edited_image, target_prompt),
            "CLIP-I": lambda: self.clip_i(source_image, edited_image),
            "LPIPS": lambda: self.lpips(source_image, edited_image),
            "DINO": lambda: self.dino(source_image, edited_image),
            "DreamSim": lambda: self.dreamsim(source_image, edited_image),
        }
        values: Dict[str, Optional[float]] = {}
        for metric in metrics:
            try:
                values[metric] = metric_fns[metric]()
            except Exception as exc:
                if not skip_failed_metrics:
                    raise
                print(f"Warning: metric {metric} failed and will be left blank: {exc}")
                values[metric] = None
        return values

    def compute_all(self, source_image, edited_image, target_prompt: str) -> Dict[str, Optional[float]]:
        return self.compute_selected(source_image, edited_image, target_prompt, list(METRIC_DIRECTIONS))


def _complete_metric_row(row: Dict[str, Any], metrics: Sequence[str]) -> bool:
    return all(str(row.get(metric, "")).strip() not in {"", "nan", "None"} for metric in metrics)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], preferred_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields: List[str] = []
    for field in preferred_fields:
        if field not in all_fields:
            all_fields.append(field)
    for row in rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics_from_run_summary(
    run_summary_csv: str | Path,
    output_root: str | Path = "outputs/flowedit_eval",
    model_name: Optional[str] = None,
    baseline_method: str = "flowedit_baseline",
    force_metrics: bool = False,
    reuse_metrics: bool = True,
    allow_missing_outputs: bool = False,
    device: Optional[str] = None,
    clip_model: str = "openai/clip-vit-base-patch32",
    dino_model: str = "facebook/dino-vitb16",
    lpips_net: str = "alex",
    lpips_resize: int = 0,
    metrics: str | Sequence[str] | None = None,
    metric_image_resolution: Optional[int] = None,
    skip_failed_metrics: bool = False,
) -> Dict[str, Path]:
    from PIL import Image

    run_rows = _read_csv_rows(Path(run_summary_csv))
    if not run_rows:
        raise RuntimeError(f"No rows found in run summary: {run_summary_csv}")

    selected_metrics = parse_metric_names(metrics)
    if not selected_metrics:
        raise ValueError("At least one metric must be selected.")

    inferred_model = model_name or run_rows[0].get("model_label") or run_rows[0].get("model_type", "model")
    model_label = model_slug(inferred_model)
    model_root = Path(output_root) / model_label
    metric_computer: Optional[MetricComputer] = None
    all_metric_rows: List[Dict[str, Any]] = []

    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in run_rows:
        method = row.get("method") or row.get("method_name") or method_name_from_solver(row.get("solver_type", ""))
        setting_id = row.get("setting_id") or "default"
        grouped.setdefault((method, setting_id), []).append(row)

    preferred_pair_fields = [
        "sample_id",
        "source_image_path",
        "source_prompt",
        "target_prompt",
        "method",
        "method_family",
        "setting_id",
        "model",
        "output_image",
        "CLIP-T",
        "CLIP-I",
        "LPIPS",
        "DINO",
        "DreamSim",
        "NFE",
        "runtime",
        "cached_generation",
    ]

    for (method, setting_id), rows in grouped.items():
        sample_ids = {row.get("sample_id", "") for row in rows}
        paths = build_eval_output_paths(model_root.parent, model_label, method, "placeholder", setting_id)
        method_root = paths["method_root"]
        per_pair_csv = method_root / "metrics_per_pair.csv"
        existing_rows = _read_csv_rows(per_pair_csv) if reuse_metrics and not force_metrics else []
        existing_by_sample = {
            row.get("sample_id", ""): row for row in existing_rows if row.get("sample_id", "") in sample_ids
        }
        method_metric_rows: List[Dict[str, Any]] = []

        for run_row in rows:
            sample_id = run_row.get("sample_id") or f"{run_row.get('source_image', 'sample')}__{run_row.get('target_index', '0')}"
            cached_metric = existing_by_sample.get(sample_id)
            if cached_metric and _complete_metric_row(cached_metric, selected_metrics):
                method_metric_rows.append(cached_metric)
                all_metric_rows.append(cached_metric)
                continue

            output_image = run_row.get("output_image", "")
            if not output_image:
                output_dir = Path(run_row.get("output_dir", ""))
                candidates = sorted(output_dir.glob("output_solver_*.png"), key=lambda p: p.stat().st_mtime)
                output_image = str(candidates[-1]) if candidates else ""
            if not output_image or not Path(output_image).exists():
                message = f"Missing edited image for sample {sample_id}: {output_image}"
                if allow_missing_outputs:
                    print(f"Skip: {message}")
                    continue
                raise FileNotFoundError(message)

            if metric_computer is None:
                metric_computer = MetricComputer(
                    device=device,
                    clip_model=clip_model,
                    dino_model=dino_model,
                    lpips_net=lpips_net,
                    lpips_resize=lpips_resize,
                )

            source_image_path = run_row.get("source_image_path") or run_row.get("source_image")
            source_image = Image.open(source_image_path).convert("RGB")
            edited_image = Image.open(output_image).convert("RGB")
            source_image = resize_longest_side(source_image, metric_image_resolution)
            edited_image = resize_longest_side(edited_image, metric_image_resolution)
            target_prompt = run_row.get("target_prompt", "")
            print(f"Metrics for {method}/{sample_id}: {', '.join(selected_metrics)}")
            metric_values = metric_computer.compute_selected(
                source_image,
                edited_image,
                target_prompt,
                selected_metrics,
                skip_failed_metrics=skip_failed_metrics,
            )

            metric_row = {
                "sample_id": sample_id,
                "source_image_path": source_image_path,
                "source_prompt": run_row.get("source_prompt", ""),
                "target_prompt": target_prompt,
                "method": method,
                "method_family": run_row.get("method_family", method),
                "setting_id": setting_id,
                "model": model_label,
                "output_image": output_image,
                "NFE": run_row.get("actual_nfe") or run_row.get("NFE") or run_row.get("estimated_nfe", ""),
                "runtime": run_row.get("elapsed_seconds") or run_row.get("runtime", ""),
                "cached_generation": run_row.get("cached_generation", ""),
            }
            for metric in METRIC_DIRECTIONS:
                value = metric_values.get(metric)
                metric_row[metric] = "" if value is None else f"{value:.6f}"
            method_metric_rows.append(metric_row)
            all_metric_rows.append(metric_row)

        method_metric_rows.sort(key=lambda row: row.get("sample_id", ""))
        _write_csv_rows(per_pair_csv, method_metric_rows, preferred_pair_fields)
        print(f"Wrote per-pair metrics: {per_pair_csv}")

    all_metric_rows.sort(key=lambda row: (row.get("method", ""), row.get("setting_id", ""), row.get("sample_id", "")))
    combined_csv = model_root / "all_metrics_per_pair.csv"
    _write_csv_rows(combined_csv, all_metric_rows, preferred_pair_fields)

    summary_csv = write_summary_reports(
        all_metric_rows,
        model_root=model_root,
        baseline_method=baseline_method,
    )
    plot_path = write_clip_lpips_plot(summary_csv, model_root=model_root)
    return {
        "combined_per_pair_csv": combined_csv,
        "summary_csv": summary_csv,
        "clip_lpips_plot": plot_path,
    }


def write_summary_reports(
    metric_rows: List[Dict[str, Any]],
    model_root: str | Path,
    baseline_method: str = "flowedit_baseline",
) -> Path:
    model_root = Path(model_root)
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in metric_rows:
        groups.setdefault((row.get("method", ""), row.get("setting_id", "default")), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (method, setting_id), rows in groups.items():
        summary: Dict[str, Any] = {
            "method": method,
            "setting_id": setting_id,
            "num_samples": len(rows),
        }
        for metric in METRIC_DIRECTIONS:
            vals = [_to_float(row.get(metric)) for row in rows]
            vals = [val for val in vals if not math.isnan(val)]
            summary[f"{metric}_mean"] = f"{(sum(vals) / len(vals)):.6f}" if vals else ""
        for src_col, out_col in [("NFE", "NFE_mean"), ("runtime", "runtime_mean")]:
            vals = [_to_float(row.get(src_col)) for row in rows]
            vals = [val for val in vals if not math.isnan(val)]
            summary[out_col] = f"{(sum(vals) / len(vals)):.3f}" if vals else ""
        summary_rows.append(summary)

    baseline_rows = [row for row in summary_rows if row["method"] == baseline_method]
    baseline = baseline_rows[0] if baseline_rows else None
    if baseline:
        for row in summary_rows:
            for metric, direction in METRIC_DIRECTIONS.items():
                base_val = _to_float(baseline.get(f"{metric}_mean"))
                val = _to_float(row.get(f"{metric}_mean"))
                if math.isnan(base_val) or math.isnan(val):
                    row[f"{metric}_delta_vs_{baseline_method}"] = ""
                elif direction == "higher":
                    row[f"{metric}_delta_vs_{baseline_method}"] = f"{(val - base_val):.6f}"
                else:
                    row[f"{metric}_delta_vs_{baseline_method}"] = f"{(base_val - val):.6f}"

    preferred_summary_fields = [
        "method",
        "setting_id",
        "num_samples",
        "CLIP-T_mean",
        "CLIP-I_mean",
        "LPIPS_mean",
        "DINO_mean",
        "DreamSim_mean",
        "NFE_mean",
        "runtime_mean",
        "CLIP-T_delta_vs_flowedit_baseline",
        "CLIP-I_delta_vs_flowedit_baseline",
        "LPIPS_delta_vs_flowedit_baseline",
        "DINO_delta_vs_flowedit_baseline",
        "DreamSim_delta_vs_flowedit_baseline",
    ]
    summary_rows.sort(key=lambda row: (row.get("method") != baseline_method, row.get("method", ""), row.get("setting_id", "")))
    summary_csv = model_root / "summary_metrics.csv"
    _write_csv_rows(summary_csv, summary_rows, preferred_summary_fields)
    print(f"Wrote summary metrics: {summary_csv}")

    delta_rows = [
        {key: value for key, value in row.items() if key in {"method", "setting_id"} or "_delta_vs_" in key}
        for row in summary_rows
    ]
    _write_csv_rows(model_root / "delta_vs_flowedit_baseline.csv", delta_rows, [])
    write_win_rate_report(metric_rows, model_root=model_root, baseline_method=baseline_method)
    write_html_table(summary_csv, model_root / "summary_table.html")
    return summary_csv


def write_win_rate_report(
    metric_rows: List[Dict[str, Any]],
    model_root: str | Path,
    baseline_method: str = "flowedit_baseline",
) -> Path:
    model_root = Path(model_root)
    baseline_by_sample = {
        row.get("sample_id", ""): row
        for row in metric_rows
        if row.get("method") == baseline_method and row.get("setting_id", "default") == "default"
    }
    rows_out: List[Dict[str, Any]] = []
    method_settings = sorted(
        {
            (row.get("method", ""), row.get("setting_id", "default"))
            for row in metric_rows
            if row.get("method") != baseline_method
        }
    )
    for method, setting_id in method_settings:
        method_rows = [
            row
            for row in metric_rows
            if row.get("method") == method and row.get("setting_id", "default") == setting_id
        ]
        out: Dict[str, Any] = {"method": method, "setting_id": setting_id, "num_comparable_samples": 0}
        wins = {metric: 0 for metric in METRIC_DIRECTIONS}
        metric_counts = {metric: 0 for metric in METRIC_DIRECTIONS}
        comparable = 0
        for row in method_rows:
            baseline = baseline_by_sample.get(row.get("sample_id", ""))
            if not baseline:
                continue
            comparable += 1
            for metric, direction in METRIC_DIRECTIONS.items():
                val = _to_float(row.get(metric))
                base_val = _to_float(baseline.get(metric))
                if math.isnan(val) or math.isnan(base_val):
                    continue
                metric_counts[metric] += 1
                if (direction == "higher" and val > base_val) or (direction == "lower" and val < base_val):
                    wins[metric] += 1
        out["num_comparable_samples"] = comparable
        for metric in METRIC_DIRECTIONS:
            count = metric_counts[metric]
            out[f"{metric}_win_rate_vs_{baseline_method}"] = f"{(wins[metric] / count):.6f}" if count else ""
            out[f"{metric}_num_valid_pairs"] = count
        rows_out.append(out)

    path = model_root / "win_rate_vs_flowedit_baseline.csv"
    _write_csv_rows(path, rows_out, [])
    print(f"Wrote win-rate table: {path}")
    return path


def write_clip_lpips_plot(summary_csv: str | Path, model_root: str | Path) -> Path:
    import pandas as pd

    model_root = Path(model_root)
    plots_dir = model_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    png_path = plots_dir / "clip_t_vs_lpips.png"
    html_path = plots_dir / "clip_t_vs_lpips.html"

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for method, group in summary.groupby("method", sort=False):
            group = group.copy()
            group["CLIP-T_mean"] = pd.to_numeric(group["CLIP-T_mean"], errors="coerce")
            group["LPIPS_mean"] = pd.to_numeric(group["LPIPS_mean"], errors="coerce")
            group = group.dropna(subset=["CLIP-T_mean", "LPIPS_mean"])
            if group.empty:
                continue
            ax.plot(group["CLIP-T_mean"], group["LPIPS_mean"], marker="o", label=method)
            for _, row in group.iterrows():
                ax.annotate(str(row.get("setting_id", "")), (row["CLIP-T_mean"], row["LPIPS_mean"]), fontsize=8)
        ax.set_xlabel("CLIP-T (higher is better)")
        ax.set_ylabel("LPIPS (lower is better)")
        ax.set_title("FlowEdit evaluation: CLIP-T vs LPIPS")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Wrote CLIP-T vs LPIPS plot: {png_path}")
    except Exception as exc:
        print(f"Could not write matplotlib plot ({exc}).")

    try:
        import plotly.express as px

        fig = px.line(
            summary,
            x="CLIP-T_mean",
            y="LPIPS_mean",
            color="method",
            markers=True,
            text="setting_id",
            title="FlowEdit evaluation: CLIP-T vs LPIPS",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="CLIP-T (higher is better)", yaxis_title="LPIPS (lower is better)")
        fig.write_html(html_path)
        print(f"Wrote interactive CLIP-T vs LPIPS plot: {html_path}")
    except Exception as exc:
        print(f"Could not write Plotly plot ({exc}).")

    return png_path


def write_html_table(csv_path: str | Path, html_path: str | Path) -> None:
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(df.to_html(index=False), encoding="utf-8")
        print(f"Wrote summary HTML table: {html_path}")
    except Exception as exc:
        print(f"Could not write HTML table ({exc}).")


def inspect_dataset_command(args: argparse.Namespace) -> None:
    samples = load_eval_samples(
        args.dataset_yaml,
        sample_limit=args.sample_limit,
        sample_offset=args.sample_offset,
        repo_root=args.repo_root,
        data_images_dir=args.data_images_dir,
    )
    full_samples = load_eval_samples(
        args.dataset_yaml,
        repo_root=args.repo_root,
        data_images_dir=args.data_images_dir,
    )
    image_count = len({sample.image_id for sample in full_samples})
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Resolved images: {image_count}")
    print(f"Total image-prompt pairs: {len(full_samples)}")
    print(f"Selected pairs: {len(samples)}")
    for sample in samples[: args.preview]:
        print(f"{sample.sample_id}: {sample.source_image_path} -> {sample.target_code}")


def write_config_command(args: argparse.Namespace) -> None:
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    bridge_settings = parse_bridge_settings(args.bridge_settings_json)
    if bridge_settings is None and any(
        value is not None
        for value in (args.bridge_lambda, args.bridge_gamma, args.bridge_weight, args.bridge_enable_below_t)
    ):
        bridge_setting = {
            "pc_guidance_lambda": args.bridge_lambda if args.bridge_lambda is not None else 1.0,
            "pc_guidance_gamma": args.bridge_gamma if args.bridge_gamma is not None else 1.0,
            "pc_guidance_weight": args.bridge_weight if args.bridge_weight is not None else 1.0,
            "pc_enable_below_t": args.bridge_enable_below_t if args.bridge_enable_below_t is not None else 1.0,
        }
        if args.bridge_setting_id:
            bridge_setting["setting_id"] = args.bridge_setting_id
        bridge_settings = [bridge_setting]
    experiments = write_experiment_yaml(
        output_yaml=args.output_yaml,
        model_name=args.model_name,
        dataset_yaml=args.dataset_yaml,
        methods=methods,
        budget_mode=args.budget_mode,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        bridge_settings=bridge_settings,
    )
    print(f"Wrote experiment config: {args.output_yaml}")
    for exp in experiments:
        print(
            f"{exp['method_name']}/{exp.get('setting_id', 'default')}: "
            f"{exp['solver_type']} T={exp['T_steps']} n_max={exp['n_max']} "
            f"src_cfg={exp['src_guidance_scale']} tar_cfg={exp['tar_guidance_scale']}"
        )


def metrics_command(args: argparse.Namespace) -> None:
    outputs = compute_metrics_from_run_summary(
        run_summary_csv=args.run_summary_csv,
        output_root=args.output_root,
        model_name=args.model_name,
        baseline_method=args.baseline_method,
        force_metrics=args.force_metrics,
        reuse_metrics=not args.no_reuse_metrics,
        allow_missing_outputs=args.allow_missing_outputs,
        device=args.device,
        clip_model=args.clip_model,
        dino_model=args.dino_model,
        lpips_net=args.lpips_net,
        lpips_resize=args.lpips_resize,
        metrics=args.metrics,
        metric_image_resolution=args.metric_image_resolution,
        skip_failed_metrics=args.skip_failed_metrics,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowEdit full evaluation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset")
    inspect_parser.add_argument("--dataset_yaml", default="Data/flowedit.yaml")
    inspect_parser.add_argument("--repo_root", default=".")
    inspect_parser.add_argument("--data_images_dir", default="Data/Images")
    inspect_parser.add_argument("--sample_limit", type=int, default=None)
    inspect_parser.add_argument("--sample_offset", type=int, default=0)
    inspect_parser.add_argument("--preview", type=int, default=10)
    inspect_parser.set_defaults(func=inspect_dataset_command)

    config_parser = subparsers.add_parser("write-config")
    config_parser.add_argument("--model_name", choices=("sd3", "flux"), default="sd3")
    config_parser.add_argument("--dataset_yaml", default="Data/flowedit.yaml")
    config_parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    config_parser.add_argument("--budget_mode", choices=("paper", "same_nfe"), default="paper")
    config_parser.add_argument("--src_guidance_scale", type=float, default=None)
    config_parser.add_argument("--tar_guidance_scale", type=float, default=None)
    config_parser.add_argument("--bridge_lambda", type=float, default=None)
    config_parser.add_argument("--bridge_gamma", type=float, default=None)
    config_parser.add_argument("--bridge_weight", type=float, default=None)
    config_parser.add_argument("--bridge_enable_below_t", type=float, default=None)
    config_parser.add_argument("--bridge_setting_id", default=None)
    config_parser.add_argument(
        "--bridge_settings_json",
        default=None,
        help=(
            "JSON object/list with bridge setting_id, pc_guidance_lambda, pc_guidance_gamma, "
            "pc_guidance_weight, and optional per-setting overrides."
        ),
    )
    config_parser.add_argument("--output_yaml", required=True)
    config_parser.set_defaults(func=write_config_command)

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--run_summary_csv", required=True)
    metrics_parser.add_argument("--output_root", default="outputs/flowedit_eval")
    metrics_parser.add_argument("--model_name", default=None)
    metrics_parser.add_argument("--baseline_method", default="flowedit_baseline")
    metrics_parser.add_argument("--force_metrics", action="store_true")
    metrics_parser.add_argument("--no_reuse_metrics", action="store_true")
    metrics_parser.add_argument("--allow_missing_outputs", action="store_true")
    metrics_parser.add_argument("--device", default=None)
    metrics_parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    metrics_parser.add_argument("--dino_model", default="facebook/dino-vitb16")
    metrics_parser.add_argument("--lpips_net", default="alex")
    metrics_parser.add_argument("--lpips_resize", type=int, default=0)
    metrics_parser.add_argument(
        "--metrics",
        default=",".join(METRIC_DIRECTIONS),
        help="Comma-separated metrics to compute. Available: CLIP-T,CLIP-I,LPIPS,DINO,DreamSim.",
    )
    metrics_parser.add_argument(
        "--metric_image_resolution",
        type=int,
        default=None,
        help="Resize source and edited images' longest side before metrics. Useful for Colab smoke tests.",
    )
    metrics_parser.add_argument(
        "--skip_failed_metrics",
        action="store_true",
        help="Leave a metric blank and keep going if an optional metric dependency fails.",
    )
    metrics_parser.set_defaults(func=metrics_command)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
