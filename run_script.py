import argparse
import csv
import gc
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from diffusers import FluxPipeline
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download, login, whoami
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from PIL import Image

from FlowEdit_utils import FlowEditSD3, FlowEditFLUX, normalize_solver_type
from flowedit_eval import (
    build_eval_output_paths,
    load_eval_samples,
    method_name_from_solver,
    model_slug,
    read_json,
    write_json,
)


HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def estimate_flowedit_nfe(T_steps, n_min, n_max, n_avg, solver_type):
    edit_steps = max(min(n_max, T_steps) - max(n_min, 0), 0)
    final_steps = min(max(n_min, 0), T_steps)
    edit_calls_per_step = 2 if solver_type in {
        "midpoint",
        "flowedit_pc_additive",
        "flowedit_pc",
        "flowedit_pc_interpolate",
        "flowedit_pc_interp",
        "flowedit_pc_interpolation",
        "flowedit_cfg_like_interpolate",
        "flowedit_cfg_like_interp",
        "flowedit_cfg_interpolate",
        "flowedit_cfg_interp",
        "flowedit_bridge_interpolate",
        "flowedit_bridge_interp",
        "flowedit_bridge_directional",
        "flowedit_bridge_direction",
        "flowedit_bridge_dir",
        "flowedit_cfgpp_no_first_term",
        "flowedit_cfgpp_no_first",
        "flowedit_remove_cfgpp_first_term",
    } else 1
    final_calls_per_step = 2 if solver_type == "midpoint" else 1
    return edit_steps * n_avg * edit_calls_per_step + final_steps * final_calls_per_step


CACHE_MATCH_FIELDS = (
    "model_id",
    "method",
    "setting_id",
    "solver_type",
    "sample_id",
    "image_resolution",
    "source_prompt",
    "target_prompt",
    "T_steps",
    "n_avg",
    "src_guidance_scale",
    "tar_guidance_scale",
    "n_min",
    "n_max",
    "pc_guidance_lambda",
    "pc_guidance_gamma",
    "pc_enable_below_t",
    "pc_guidance_weight",
    "seed",
)


def _cache_values_equal(left, right) -> bool:
    if left is None:
        left = ""
    if right is None:
        right = ""
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return str(left) == str(right)


def cache_mismatch_reason(cached_metadata, expected_metadata):
    if not cached_metadata:
        return "missing metadata"
    for field in CACHE_MATCH_FIELDS:
        if not _cache_values_equal(cached_metadata.get(field), expected_metadata.get(field)):
            return f"{field}: cached={cached_metadata.get(field)!r}, expected={expected_metadata.get(field)!r}"
    return None


def describe_flowedit_solver(solver_type: str):
    solver_type = normalize_solver_type(solver_type)
    descriptions = {
        "euler": {
            "theory_family": "original_flowedit",
            "theory_formula": "G_t",
            "midpoint_space": "none",
            "correction_mode": "none",
        },
        "midpoint": {
            "theory_family": "generic_midpoint",
            "theory_formula": "G_mid(z_t + h/2 G_t)",
            "midpoint_space": "edit_latent_generic",
            "correction_mode": "replace_with_midpoint",
        },
        "flowedit_pc_additive": {
            "theory_family": "branch_midpoint_additive",
            "theory_formula": "G_t + alpha(t) G_mid^branch",
            "midpoint_space": "source_target_branches",
            "correction_mode": "additive",
        },
        "flowedit_pc_interpolate": {
            "theory_family": "branch_midpoint_interpolate",
            "theory_formula": "(1-alpha(t)) G_t + alpha(t) G_mid^branch",
            "midpoint_space": "source_target_branches",
            "correction_mode": "interpolate",
        },
        "flowedit_cfg_like_interpolate": {
            "theory_family": "branch_midpoint_cfg_like",
            "theory_formula": "(1-alpha(t)) G_t + alpha(t) G_mid^branch",
            "midpoint_space": "source_target_branches",
            "correction_mode": "interpolate",
        },
        "flowedit_cfgpp_no_first_term": {
            "theory_family": "branch_midpoint_cfgpp_no_first_term",
            "theory_formula": "alpha(t) G_mid^branch",
            "midpoint_space": "source_target_branches",
            "correction_mode": "remove_first_term",
        },
        "flowedit_bridge_interpolate": {
            "theory_family": "bridge_midpoint_interpolate",
            "theory_formula": "(1-alpha(t)) G_t + alpha(t) G_mid^bridge",
            "midpoint_space": "shared_noise_bridge",
            "correction_mode": "interpolate",
        },
        "flowedit_bridge_directional": {
            "theory_family": "bridge_midpoint_directional",
            "theory_formula": "dir(hat G_t)=blend(dir G_t, dir G_mid^bridge), ||hat G_t||=||G_t||",
            "midpoint_space": "shared_noise_bridge",
            "correction_mode": "direction_preserving",
        },
    }
    return descriptions[solver_type] | {"normalized_solver_type": solver_type}


def select_runtime(device_number: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_number}"), "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch.float16
    return torch.device("cpu"), "cpu", torch.float32


def inference_context(runtime_name: str):
    if runtime_name == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


def default_model_id(model_type: str) -> str:
    if model_type == "FLUX":
        return "black-forest-labs/FLUX.1-dev"
    if model_type == "SD3":
        return "stabilityai/stable-diffusion-3-medium-diffusers"
    raise NotImplementedError(f"Model type {model_type} not implemented")


def resolve_hf_token() -> Tuple[Optional[str], Optional[str]]:
    for key in HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value, key
    return None, None


def maybe_login_huggingface(token: Optional[str]) -> Optional[str]:
    if not token:
        print("No Hugging Face token detected in environment. Public downloads only.")
        return None

    try:
        login(token=token, add_to_git_credential=False, new_session=False)
        user_info = whoami(token=token)
        hf_user = user_info.get("name") or user_info.get("fullname")
        if hf_user:
            print(f"Hugging Face access token accepted for user: {hf_user}")
        else:
            print("Hugging Face access token accepted.")
        return hf_user
    except Exception as exc:
        print(f"Warning: Hugging Face login could not be verified ({exc}). Continuing with direct token usage.")
        return None


def ensure_model_access(model_id: str, token: Optional[str], cache_dir: str):
    try:
        probe_path = hf_hub_download(
            repo_id=model_id,
            filename="model_index.json",
            token=token,
            cache_dir=cache_dir,
        )
        print(f"Model access check passed for {model_id}. Cached model_index.json at: {probe_path}")
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Cannot access gated model '{model_id}'. Visit https://huggingface.co/{model_id}, "
            f"accept the access terms, and set one of {HF_TOKEN_ENV_KEYS} before rerunning."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"Model repository '{model_id}' was not found. Check the model_id in your experiment YAML."
        ) from exc
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise RuntimeError(
                f"Access denied while checking '{model_id}' (HTTP {status_code}). "
                f"Set a valid Hugging Face token and make sure that account can access https://huggingface.co/{model_id}."
            ) from exc
        raise RuntimeError(f"Failed to verify access to '{model_id}': {exc}") from exc


def resolve_pipeline_load_mode(runtime_name: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        if runtime_name == "cuda":
            # Free Colab T4 runs are much more stable with CPU offload.
            return "model_cpu_offload"
        return "to_device"

    if runtime_name != "cuda" and requested_mode in {"model_cpu_offload", "sequential_cpu_offload"}:
        print(
            f"Requested pipeline load mode '{requested_mode}' requires CUDA. "
            "Falling back to 'to_device' for this runtime."
        )
        return "to_device"

    return requested_mode


def build_pipeline(model_type: str, model_id: str, model_dtype: torch.dtype, token: Optional[str]):
    load_kwargs = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if token:
        load_kwargs["token"] = token

    if model_type == "FLUX":
        return FluxPipeline.from_pretrained(model_id, **load_kwargs)
    if model_type == "SD3":
        return StableDiffusion3Pipeline.from_pretrained(model_id, **load_kwargs)
    raise NotImplementedError(f"Model type {model_type} not implemented")


def configure_pipeline_runtime(pipe, runtime_name: str, device: torch.device, device_number: int, load_mode: str):
    if runtime_name == "cuda":
        pipe.enable_attention_slicing("auto")
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        if load_mode == "model_cpu_offload":
            pipe.enable_model_cpu_offload(gpu_id=device_number, device="cuda")
            return pipe
        if load_mode == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload(gpu_id=device_number, device="cuda")
            return pipe

    pipe = pipe.to(device)
    return pipe


def resize_image_for_eval(image: Image.Image, image_resolution: Optional[int]) -> Image.Image:
    if image_resolution is None or image_resolution <= 0:
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side == image_resolution:
        return image
    scale = image_resolution / longest_side
    new_width = max(16, int(round(width * scale)))
    new_height = max(16, int(round(height * scale)))
    new_width -= new_width % 16
    new_height -= new_height % 16
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    return image.resize((new_width, new_height), resample=resample)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--exp_yaml", type=str, default="FLUX_exp.yaml", help="experiment yaml file")
    parser.add_argument(
        "--pipeline_load_mode",
        type=str,
        default="auto",
        choices=("auto", "to_device", "model_cpu_offload", "sequential_cpu_offload"),
        help="How to place pipeline weights at runtime. 'auto' prefers CPU offload on CUDA for Colab/T4 stability.",
    )
    parser.add_argument(
        "--skip_model_access_check",
        action="store_true",
        help="Skip the Hugging Face file-access preflight. Useful only when you already have the model cached.",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Stop after model access, loading, and runtime placement checks succeed.",
    )
    parser.add_argument(
        "--dataset_yaml",
        type=str,
        default=None,
        help="Override the dataset YAML for every experiment. Supports Data/flowedit.yaml and legacy edits YAML files.",
    )
    parser.add_argument(
        "--data_images_dir",
        type=str,
        default="Data/Images",
        help="Directory used to resolve FlowEdit released-dataset image paths such as flowedit_data/name.png.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit the flattened image-prompt pairs. Omit for the full dataset.",
    )
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help="Skip this many flattened image-prompt pairs before applying sample_limit.",
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=None,
        help="Resize the source image's longest side before editing. Omit for native/paper resolution.",
    )
    parser.add_argument(
        "--eval_output_root",
        type=str,
        default=None,
        help="Use the full-eval output layout under this root, e.g. outputs/flowedit_eval.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Regenerate edited images even when cached full-eval outputs already exist.",
    )
    parser.add_argument(
        "--run_summary_csv",
        type=str,
        default=None,
        help="Where to write the generation/runtime CSV. Defaults to outputs/run_summary.csv or <eval_output_root>/<model>/run_summary.csv.",
    )

    args = parser.parse_args()

    # set device
    device_number = args.device_number
    device, runtime_name, model_dtype = select_runtime(device_number)

    # load exp yaml file to dict
    exp_yaml = args.exp_yaml
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    hf_home = os.environ.setdefault("HF_HOME", os.path.join(repo_root, ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    for cache_dir in {hf_home, os.environ["HF_HUB_CACHE"], os.environ["TRANSFORMERS_CACHE"]}:
        os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    hf_token, hf_token_source = resolve_hf_token()
    hf_user = maybe_login_huggingface(hf_token)

    print(f"Runtime device: {device} ({runtime_name}), model dtype: {model_dtype}")
    if runtime_name != "cuda":
        print(
            "Warning: CUDA is not available. Full SD3/FLUX checkpoints may be too slow or exceed memory "
            "on CPU/MPS-only machines."
        )
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device_number)
        total_vram_gb = torch.cuda.get_device_properties(device_number).total_memory / (1024 ** 3)
        print(f"CUDA device {device_number}: {gpu_name} ({total_vram_gb:.1f} GiB VRAM)")

    model_type = exp_configs[0]["model_type"] # currently only one model type per run
    model_label = model_slug(model_type)
    model_id = exp_configs[0].get("model_id", default_model_id(model_type))
    if args.run_summary_csv:
        summary_path = args.run_summary_csv
    elif args.eval_output_root:
        summary_path = os.path.join(args.eval_output_root, model_label, "run_summary.csv")
    else:
        summary_path = "outputs/run_summary.csv"
    requested_load_mode = args.pipeline_load_mode
    if requested_load_mode == "auto":
        requested_load_mode = exp_configs[0].get("pipeline_load_mode", "auto")
    pipeline_load_mode = resolve_pipeline_load_mode(runtime_name, requested_load_mode)
    print(f"Pipeline load mode: {pipeline_load_mode} (requested: {requested_load_mode})")
    if hf_token_source:
        print(f"Hugging Face token source: {hf_token_source}")

    if not args.skip_model_access_check:
        ensure_model_access(model_id, hf_token, os.environ["HF_HUB_CACHE"])
    else:
        print(f"Skipping model access check for {model_id}.")

    pipe = build_pipeline(model_type, model_id, model_dtype, hf_token)
    pipe = configure_pipeline_runtime(pipe, runtime_name, device, device_number, pipeline_load_mode)
    print(f"Pipeline load complete for {model_type} using model '{model_id}'.")
    
    scheduler = pipe.scheduler
    run_summaries = []

    if args.preflight_only:
        print("Preflight completed successfully.")
        raise SystemExit(0)

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        solver_type = exp_dict.get("solver_type", "euler")
        solver_info = describe_flowedit_solver(solver_type)
        method = exp_dict.get("method_name") or method_name_from_solver(solver_type)
        method_family = exp_dict.get("method_family", method)
        setting_id = exp_dict.get("setting_id", "default")
        pc_guidance_lambda = exp_dict.get("pc_guidance_lambda", 1.0)
        pc_guidance_gamma = exp_dict.get("pc_guidance_gamma", 1.0)
        pc_enable_below_t = exp_dict.get("pc_enable_below_t", 1.0)
        pc_guidance_weight = exp_dict.get("pc_guidance_weight", 1.0)
        estimated_nfe = estimate_flowedit_nfe(T_steps, n_min, n_max, n_avg, solver_type)
        seed = exp_dict["seed"]
        dataset_yaml = args.dataset_yaml or exp_dict["dataset_yaml"]
        samples = load_eval_samples(
            dataset_yaml,
            sample_limit=args.sample_limit,
            sample_offset=args.sample_offset,
            repo_root=repo_root,
            data_images_dir=args.data_images_dir,
        )
        print(
            f"Experiment {exp_name}: method={method}, solver={solver_type}, "
            f"samples={len(samples)}, estimated NFE={estimated_nfe}"
        )

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for sample in samples:
            src_prompt = sample.source_prompt
            tar_prompt = sample.target_prompt
            negative_prompt = sample.negative_prompt
            image_src_path = sample.source_image_path

            if args.eval_output_root:
                paths = build_eval_output_paths(
                    args.eval_output_root,
                    model_label,
                    method,
                    sample.sample_id,
                    setting_id,
                )
                output_image_path = paths["output_image"]
                metadata_json_path = paths["metadata_json"]
                prompts_txt_path = paths["prompts_txt"]
                save_dir = str(paths["image_dir"])
            else:
                save_dir = f"outputs/{exp_name}/{model_type}/src_{sample.image_id}/tar_{sample.target_index}"
                filename = (
                    f"output_solver_{solver_type}_T_steps_{T_steps}_n_avg_{n_avg}_"
                    f"cfg_enc_{src_guidance_scale}_cfg_dec{tar_guidance_scale}_"
                    f"n_min_{n_min}_n_max_{n_max}_seed{seed}.png"
                )
                output_image_path = Path(save_dir) / filename
                metadata_json_path = Path(save_dir) / "metadata.json"
                prompts_txt_path = Path(save_dir) / "prompts.txt"

            output_image_path = Path(output_image_path)
            cached_generation = False
            cached_metadata = read_json(metadata_json_path)
            expected_cache_metadata = {
                "model_id": model_id,
                "method": method,
                "setting_id": setting_id,
                "solver_type": solver_type,
                "sample_id": sample.sample_id,
                "image_resolution": args.image_resolution or "",
                "source_prompt": src_prompt,
                "target_prompt": tar_prompt,
                "T_steps": T_steps,
                "n_avg": n_avg,
                "src_guidance_scale": src_guidance_scale,
                "tar_guidance_scale": tar_guidance_scale,
                "n_min": n_min,
                "n_max": n_max,
                "pc_guidance_lambda": pc_guidance_lambda,
                "pc_guidance_gamma": pc_guidance_gamma,
                "pc_enable_below_t": pc_enable_below_t,
                "pc_guidance_weight": pc_guidance_weight,
                "seed": seed,
            }
            cache_reason = cache_mismatch_reason(cached_metadata, expected_cache_metadata)
            actual_nfe = cached_metadata.get("actual_nfe", estimated_nfe)
            elapsed_seconds = float(cached_metadata.get("elapsed_seconds", 0.0) or 0.0)

            if args.eval_output_root and output_image_path.exists() and not args.force_rerun and cache_reason is None:
                cached_generation = True
                print(f"Skip cached output: {output_image_path}")
            else:
                if args.eval_output_root and output_image_path.exists() and not args.force_rerun and cache_reason:
                    print(f"Regenerate stale cached output ({cache_reason}): {output_image_path}")
                start_time = time.perf_counter()

                image = Image.open(image_src_path).convert("RGB")
                image = resize_image_for_eval(image, args.image_resolution)
                # Crop to dimensions divisible by 16 to avoid VAE resizing issues.
                image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                image_src = pipe.image_processor.preprocess(image)
                image_src = image_src.to(device=device, dtype=model_dtype)
                with inference_context(runtime_name), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                x0_src = x0_src.to(device)

                if model_type == 'SD3':
                    x0_tar, flowedit_stats = FlowEditSD3(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,
                                                            solver_type,
                                                            pc_guidance_lambda=pc_guidance_lambda,
                                                            pc_guidance_gamma=pc_guidance_gamma,
                                                            pc_enable_below_t=pc_enable_below_t,
                                                            pc_guidance_weight=pc_guidance_weight,
                                                            return_stats=True,)
                    
                elif model_type == 'FLUX':
                    x0_tar, flowedit_stats = FlowEditFLUX(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,
                                                            solver_type,
                                                            pc_guidance_lambda=pc_guidance_lambda,
                                                            pc_guidance_gamma=pc_guidance_gamma,
                                                            pc_enable_below_t=pc_enable_below_t,
                                                            pc_guidance_weight=pc_guidance_weight,
                                                            return_stats=True,)
                else:
                    raise NotImplementedError(f"Sampler type {model_type} not implemented")

                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with inference_context(runtime_name), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                image_tar = pipe.image_processor.postprocess(image_tar)

                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                image_tar[0].save(output_image_path)
                elapsed_seconds = time.perf_counter() - start_time
                actual_nfe = flowedit_stats.get("actual_nfe", estimated_nfe)

            run_row = {
                "exp_name": exp_name,
                "model_type": model_type,
                "model_label": model_label,
                "model_id": model_id,
                "method": method,
                "method_family": method_family,
                "setting_id": setting_id,
                "solver_type": solver_type,
                "normalized_solver_type": solver_info["normalized_solver_type"],
                "theory_family": solver_info["theory_family"],
                "theory_formula": solver_info["theory_formula"],
                "midpoint_space": solver_info["midpoint_space"],
                "correction_mode": solver_info["correction_mode"],
                "runtime_name": runtime_name,
                "pipeline_load_mode": pipeline_load_mode,
                "hf_token_source": hf_token_source or "",
                "hf_user": hf_user or "",
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "image_resolution": args.image_resolution or "",
                "source_image": image_src_path,
                "source_image_path": image_src_path,
                "source_prompt": src_prompt,
                "target_index": sample.target_index,
                "target_code": sample.target_code,
                "target_prompt": tar_prompt,
                "T_steps": T_steps,
                "n_avg": n_avg,
                "src_guidance_scale": src_guidance_scale,
                "tar_guidance_scale": tar_guidance_scale,
                "n_min": n_min,
                "n_max": n_max,
                "estimated_nfe": estimated_nfe,
                "actual_nfe": actual_nfe,
                "NFE": actual_nfe,
                "pc_guidance_lambda": pc_guidance_lambda,
                "pc_guidance_gamma": pc_guidance_gamma,
                "pc_enable_below_t": pc_enable_below_t,
                "pc_guidance_weight": pc_guidance_weight,
                "seed": seed,
                "negative_prompt": negative_prompt,
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
                "runtime": f"{elapsed_seconds:.3f}",
                "cached_generation": cached_generation,
                "output_dir": save_dir,
                "output_image": str(output_image_path),
            }
            run_summaries.append(run_row)

            write_json(metadata_json_path, run_row)
            prompts_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompts_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Sample ID: {sample.sample_id}\n")
                f.write(f"Source image: {image_src_path}\n")
                f.write(f"Image resolution: {args.image_resolution or 'native'}\n")
                f.write(f"Source prompt: {src_prompt}\n")
                f.write(f"Target prompt: {tar_prompt}\n")
                f.write(f"Negative prompt: {negative_prompt}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Model ID: {model_id}\n")
                f.write(f"Sampler type: {model_type}\n")
                f.write(f"Method: {method}\n")
                f.write(f"Solver type: {solver_type}\n")
                f.write(f"Normalized solver type: {solver_info['normalized_solver_type']}\n")
                f.write(f"Theory family: {solver_info['theory_family']}\n")
                f.write(f"Theory formula: {solver_info['theory_formula']}\n")
                f.write(f"Midpoint space: {solver_info['midpoint_space']}\n")
                f.write(f"Correction mode: {solver_info['correction_mode']}\n")
                f.write(f"Runtime: {runtime_name}\n")
                f.write(f"Pipeline load mode: {pipeline_load_mode}\n")
                f.write(f"Hugging Face token source: {hf_token_source or 'none'}\n")
                f.write(f"Hugging Face user: {hf_user or 'unknown'}\n")
                f.write(f"Estimated NFE: {estimated_nfe}\n")
                f.write(f"Actual NFE: {actual_nfe}\n")
                f.write(f"PC guidance lambda: {pc_guidance_lambda}\n")
                f.write(f"PC guidance gamma: {pc_guidance_gamma}\n")
                f.write(f"PC enable below t: {pc_enable_below_t}\n")
                f.write(f"PC guidance weight: {pc_guidance_weight}\n")
                f.write(f"Runtime seconds: {elapsed_seconds:.3f}\n")
            if runtime_name == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    if run_summaries:
        summary_dir = os.path.dirname(summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=run_summaries[0].keys())
            writer.writeheader()
            writer.writerows(run_summaries)
        print(f"Wrote runtime summary to {summary_path}")

    print("Done")

    # %%
