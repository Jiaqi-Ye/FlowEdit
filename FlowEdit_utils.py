from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps



def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def normalize_solver_type(solver_type: str) -> str:
    solver_type = solver_type.lower()
    if solver_type == "flowedit_pc":
        return "flowedit_pc_additive"
    if solver_type in {"flowedit_pc_interp", "flowedit_pc_interpolation"}:
        return "flowedit_pc_interpolate"
    if solver_type in {"flowedit_cfg_interp", "flowedit_cfg_interpolate", "flowedit_cfg_like_interp"}:
        return "flowedit_cfg_like_interpolate"
    if solver_type in {"flowedit_bridge_interp", "flowedit_bridge_interpolation"}:
        return "flowedit_bridge_interpolate"
    if solver_type in {"flowedit_bridge_dir", "flowedit_bridge_direction"}:
        return "flowedit_bridge_directional"
    if solver_type in {
        "flowedit_bridge_dir_mag",
        "flowedit_bridge_direction_magnitude",
        "flowedit_bridge_directional_magnitude",
    }:
        return "flowedit_bridge_dir_mag"
    if solver_type in {
        "flowedit_cfgpp_no_first",
        "flowedit_cfgpp_no_first_term",
        "flowedit_remove_cfgpp_first_term",
    }:
        return "flowedit_cfgpp_no_first_term"
    if solver_type in {
        "euler",
        "midpoint",
        "flowedit_pc_additive",
        "flowedit_pc_interpolate",
        "flowedit_cfg_like_interpolate",
        "flowedit_cfgpp_no_first_term",
        "flowedit_bridge_interpolate",
        "flowedit_bridge_directional",
        "flowedit_bridge_dir_mag",
    }:
        return solver_type
    raise ValueError(
        f"Unsupported solver_type: {solver_type}. "
        "Use 'euler', 'midpoint', 'flowedit_pc_additive', "
        "'flowedit_pc_interpolate', 'flowedit_cfg_like_interpolate', "
        "'flowedit_cfgpp_no_first_term', 'flowedit_bridge_interpolate', "
        "'flowedit_bridge_directional', or 'flowedit_bridge_dir_mag'."
    )


def rectified_flowedit_alpha(t, lambda_: float = 1.0, gamma: float = 1.0):
    """Time-scheduled correction weight alpha(t)=lambda*(1-t)^gamma."""
    return lambda_ * (1 - t).clamp_min(0) ** gamma


def scheduled_flowedit_alpha(t, lambda_: float = 1.0, gamma: float = 1.0, enable_below_t: float = 1.0):
    """Optionally enable the correction only in the later part of the trajectory."""
    alpha = rectified_flowedit_alpha(t, lambda_, gamma)
    threshold = torch.as_tensor(enable_below_t, device=alpha.device, dtype=alpha.dtype)
    return torch.where(t <= threshold, alpha, torch.zeros_like(alpha))


def flowedit_cfg_like_contrast(v_tar, v_src):
    """CFG-like FlowEdit field: target velocity plus negative source velocity."""
    return v_tar + (-v_src)


def blend_direction_preserve_norm(v_base, v_mid, alpha, mid_weight: float = 1.0, eps: float = 1e-6):
    """Rotate the base FlowEdit field toward the midpoint field without inflating its norm."""
    base_f32 = v_base.to(torch.float32)
    mid_f32 = (mid_weight * v_mid).to(torch.float32)

    base_flat = base_f32.reshape(base_f32.shape[0], -1)
    mid_flat = mid_f32.reshape(mid_f32.shape[0], -1)

    base_norm = torch.linalg.vector_norm(base_flat, dim=1, keepdim=True).clamp_min(eps)
    mid_norm = torch.linalg.vector_norm(mid_flat, dim=1, keepdim=True).clamp_min(eps)

    base_dir = base_flat / base_norm
    mid_dir = mid_flat / mid_norm

    alpha = torch.as_tensor(alpha, device=base_f32.device, dtype=base_f32.dtype).reshape(1, 1)
    cos_sim = (base_dir * mid_dir).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    gated_alpha = alpha * cos_sim.clamp_min(0.0)

    blend_dir = (1 - gated_alpha) * base_dir + gated_alpha * mid_dir
    blend_norm = torch.linalg.vector_norm(blend_dir, dim=1, keepdim=True).clamp_min(eps)
    blend_dir = blend_dir / blend_norm

    blended = (base_norm * blend_dir).reshape_as(base_f32)
    return blended.to(v_base.dtype)


def blend_direction_with_magnitude_recovery(
    v_base,
    v_mid,
    alpha,
    mid_weight: float = 1.0,
    magnitude_weight: float = 0.2,
    magnitude_min: float = 1.0,
    magnitude_max: float = 1.1,
    eps: float = 1e-6,
):
    """Rotate toward the midpoint field, then recover only a clamped amount of midpoint norm."""
    base_f32 = v_base.to(torch.float32)
    mid_f32 = (mid_weight * v_mid).to(torch.float32)

    base_flat = base_f32.reshape(base_f32.shape[0], -1)
    mid_flat = mid_f32.reshape(mid_f32.shape[0], -1)

    base_norm = torch.linalg.vector_norm(base_flat, dim=1, keepdim=True).clamp_min(eps)
    mid_norm = torch.linalg.vector_norm(mid_flat, dim=1, keepdim=True).clamp_min(eps)

    base_dir = base_flat / base_norm
    mid_dir = mid_flat / mid_norm

    alpha = torch.as_tensor(alpha, device=base_f32.device, dtype=base_f32.dtype).reshape(1, 1)
    cos_sim = (base_dir * mid_dir).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    alpha_dir = alpha * cos_sim.clamp_min(0.0)

    blend_dir = (1 - alpha_dir) * base_dir + alpha_dir * mid_dir
    blend_norm = torch.linalg.vector_norm(blend_dir, dim=1, keepdim=True).clamp_min(eps)
    blend_dir = blend_dir / blend_norm

    mag_alpha = (alpha * float(magnitude_weight)).clamp(0.0, 1.0)
    target_norm = (1 - mag_alpha) * base_norm + mag_alpha * mid_norm
    norm_ratio = (target_norm / base_norm).clamp(float(magnitude_min), float(magnitude_max))
    recovered_norm = base_norm * norm_ratio

    blended = (recovered_norm * blend_dir).reshape_as(base_f32)
    return blended.to(v_base.dtype)


def velocity_to_score(x, v, t, eps: float = 1e-5):
    """Convert rectified-flow velocity to score using score=(t*v-x)/(1-t)."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)
    while t.ndim < x.ndim:
        t = t.view(*t.shape, 1)
    return (t * v - x) / (1 - t).clamp_min(eps)



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred



@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    solver_type: str = "euler",
    pc_guidance_lambda: float = 1.0,
    pc_guidance_gamma: float = 1.0,
    pc_enable_below_t: float = 1.0,
    pc_guidance_weight: float = 1.0,
    pc_magnitude_weight: float = 0.2,
    pc_magnitude_min: float = 1.0,
    pc_magnitude_max: float = 1.1,
    return_stats: bool = False,):
    
    device = x_src.device
    solver_type = normalize_solver_type(solver_type)

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    actual_nfe = 0
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    def flowedit_delta(z_edit, t_unit, t_model, noises):
        nonlocal actual_nfe
        V_delta_avg = torch.zeros_like(x_src)
        for fwd_noise in noises:
            zt_src = (1-t_unit)*x_src + (t_unit)*fwd_noise
            zt_tar = z_edit + zt_src - x_src

            src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar])

            Vt_src, Vt_tar = calc_v_sd3(
                pipe,
                src_tar_latent_model_input,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                src_guidance_scale,
                tar_guidance_scale,
                t_model,
            )
            actual_nfe += 1

            V_delta_avg += (1/n_avg) * flowedit_cfg_like_contrast(Vt_tar, Vt_src)
        return V_delta_avg

    def flowedit_pc_delta(z_edit, t_unit, t_model, t_mid_unit, t_mid_model, dt, noises, combine_mode):
        nonlocal actual_nfe
        V_hat_avg = torch.zeros_like(x_src)
        alpha = scheduled_flowedit_alpha(t_unit, pc_guidance_lambda, pc_guidance_gamma, pc_enable_below_t)
        if float(alpha.max().item()) <= 0:
            return flowedit_delta(z_edit, t_unit, t_model, noises)
        for fwd_noise in noises:
            zt_src = (1-t_unit)*x_src + (t_unit)*fwd_noise
            zt_tar = z_edit + zt_src - x_src

            src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
            Vt_src, Vt_tar = calc_v_sd3(
                pipe,
                src_tar_latent_model_input,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                src_guidance_scale,
                tar_guidance_scale,
                t_model,
            )
            actual_nfe += 1

            V_delta = flowedit_cfg_like_contrast(Vt_tar, Vt_src)
            zt_src_mid = (zt_src.to(torch.float32) + 0.5 * dt * Vt_src).to(Vt_src.dtype)
            zt_tar_mid = (zt_tar.to(torch.float32) + 0.5 * dt * Vt_tar).to(Vt_tar.dtype)

            src_tar_mid_latent_model_input = torch.cat([zt_src_mid, zt_src_mid, zt_tar_mid, zt_tar_mid])
            Vt_src_mid, Vt_tar_mid = calc_v_sd3(
                pipe,
                src_tar_mid_latent_model_input,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                src_guidance_scale,
                tar_guidance_scale,
                t_mid_model,
            )
            actual_nfe += 1

            V_delta_mid = pc_guidance_weight * flowedit_cfg_like_contrast(Vt_tar_mid, Vt_src_mid)
            if combine_mode == "additive":
                V_hat = V_delta + alpha * V_delta_mid
            elif combine_mode == "interpolate":
                V_hat = (1 - alpha) * V_delta + alpha * V_delta_mid
            elif combine_mode == "cfg_like_interpolate":
                V_hat = (1 - alpha) * V_delta + alpha * V_delta_mid
            elif combine_mode == "cfgpp_no_first_term":
                V_hat = alpha * V_delta_mid
            else:
                raise ValueError(f"Unsupported FlowEdit PC combine_mode: {combine_mode}")
            V_hat_avg += (1/n_avg) * V_hat
        return V_hat_avg

    def flowedit_bridge_pc_delta(z_edit, t_unit, t_model, t_mid_unit, t_mid_model, dt, noises, correction_mode="interpolate"):
        V_delta = flowedit_delta(z_edit, t_unit, t_model, noises)
        alpha = scheduled_flowedit_alpha(t_unit, pc_guidance_lambda, pc_guidance_gamma, pc_enable_below_t)
        if float(alpha.max().item()) <= 0:
            return V_delta
        zt_mid = (z_edit.to(torch.float32) + 0.5 * dt * V_delta).to(V_delta.dtype)
        V_delta_mid = flowedit_delta(zt_mid, t_mid_unit, t_mid_model, noises)
        if correction_mode == "directional":
            return blend_direction_preserve_norm(V_delta, V_delta_mid, alpha, pc_guidance_weight)
        if correction_mode == "dir_mag":
            return blend_direction_with_magnitude_recovery(
                V_delta,
                V_delta_mid,
                alpha,
                mid_weight=pc_guidance_weight,
                magnitude_weight=pc_magnitude_weight,
                magnitude_min=pc_magnitude_min,
                magnitude_max=pc_magnitude_max,
            )
        return (1 - alpha) * V_delta + alpha * (pc_guidance_weight * V_delta_mid)

    def target_velocity(z_tar, t_model):
        nonlocal actual_nfe
        src_tar_latent_model_input = torch.cat([z_tar, z_tar, z_tar, z_tar])
        _, Vt_tar = calc_v_sd3(
            pipe,
            src_tar_latent_model_input,
            src_tar_prompt_embeds,
            src_tar_pooled_prompt_embeds,
            src_guidance_scale,
            tar_guidance_scale,
            t_model,
        )
        actual_nfe += 1
        return Vt_tar

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_next = timesteps[i+1]
            t_im1 = t_next/1000
        else:
            t_next = torch.zeros_like(t).to(t.device)
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        dt = t_im1 - t_i
        t_mid = t + 0.5 * (t_next - t)
        t_mid_i = t_mid/1000

        if T_steps - i > n_min:

            fwd_noises = [torch.randn_like(x_src).to(x_src.device) for _ in range(n_avg)]
            if solver_type in {
                "flowedit_pc_additive",
                "flowedit_pc_interpolate",
                "flowedit_cfg_like_interpolate",
                "flowedit_cfgpp_no_first_term",
            }:
                V_delta_avg = flowedit_pc_delta(
                    zt_edit,
                    t_i,
                    t,
                    t_mid_i,
                    t_mid,
                    dt,
                    fwd_noises,
                    "cfg_like_interpolate"
                    if solver_type == "flowedit_cfg_like_interpolate"
                    else "cfgpp_no_first_term"
                    if solver_type == "flowedit_cfgpp_no_first_term"
                    else "interpolate"
                    if solver_type == "flowedit_pc_interpolate"
                    else "additive",
                )
            elif solver_type in {"flowedit_bridge_interpolate", "flowedit_bridge_directional", "flowedit_bridge_dir_mag"}:
                V_delta_avg = flowedit_bridge_pc_delta(
                    zt_edit,
                    t_i,
                    t,
                    t_mid_i,
                    t_mid,
                    dt,
                    fwd_noises,
                    correction_mode="directional"
                    if solver_type == "flowedit_bridge_directional"
                    else "dir_mag"
                    if solver_type == "flowedit_bridge_dir_mag"
                    else "interpolate",
                )
            else:
                V_delta_avg = flowedit_delta(zt_edit, t_i, t, fwd_noises)

            if solver_type == "midpoint":
                zt_mid = (zt_edit.to(torch.float32) + 0.5 * dt * V_delta_avg).to(V_delta_avg.dtype)
                V_delta_avg = flowedit_delta(zt_mid, t_mid_i, t_mid, fwd_noises)

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + dt * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            Vt_tar = target_velocity(xt_tar, t)

            if solver_type == "midpoint":
                xt_mid = (xt_tar.to(torch.float32) + 0.5 * dt * Vt_tar).to(Vt_tar.dtype)
                Vt_tar = target_velocity(xt_mid, t_mid)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + dt * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)

            xt_tar = prev_sample
        
    out = zt_edit if n_min == 0 else xt_tar
    if return_stats:
        return out, {"actual_nfe": actual_nfe}
    return out



@torch.no_grad()
def FlowEditFLUX(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    solver_type: str = "euler",
    pc_guidance_lambda: float = 1.0,
    pc_guidance_gamma: float = 1.0,
    pc_enable_below_t: float = 1.0,
    pc_guidance_weight: float = 1.0,
    pc_magnitude_weight: float = 0.2,
    pc_magnitude_min: float = 1.0,
    pc_magnitude_max: float = 1.1,
    return_stats: bool = False,):

    device = x_src.device
    solver_type = normalize_solver_type(solver_type)
    orig_height, orig_width = x_src.shape[2]*pipe.vae_scale_factor//2, x_src.shape[3]*pipe.vae_scale_factor//2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size= x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None,latents=x_src)
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )
    
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    actual_nfe = 0

    
    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    def flowedit_delta(z_edit, sigma, t_model, noises):
        nonlocal actual_nfe
        V_delta_avg = torch.zeros_like(x_src_packed)
        for fwd_noise in noises:
            zt_src = (1-sigma)*x_src_packed + (sigma)*fwd_noise
            zt_tar = z_edit + zt_src - x_src_packed

            Vt_src = calc_v_flux(pipe,
                                                latents=zt_src,
                                                prompt_embeds=src_prompt_embeds,
                                                pooled_prompt_embeds=src_pooled_prompt_embeds,
                                                guidance=src_guidance,
                                                text_ids=src_text_ids,
                                                latent_image_ids=latent_src_image_ids,
                                                t=t_model)

            Vt_tar = calc_v_flux(pipe,
                                                latents=zt_tar,
                                                prompt_embeds=tar_prompt_embeds,
                                                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                                guidance=tar_guidance,
                                                text_ids=tar_text_ids,
                                                latent_image_ids=latent_tar_image_ids,
                                                t=t_model)
            actual_nfe += 1

            V_delta_avg += (1/n_avg) * flowedit_cfg_like_contrast(Vt_tar, Vt_src)
        return V_delta_avg

    def flowedit_pc_delta(z_edit, sigma, t_model, sigma_mid, t_mid_model, dt, noises, combine_mode):
        nonlocal actual_nfe
        V_hat_avg = torch.zeros_like(x_src_packed)
        alpha = scheduled_flowedit_alpha(sigma, pc_guidance_lambda, pc_guidance_gamma, pc_enable_below_t)
        if float(alpha.max().item()) <= 0:
            return flowedit_delta(z_edit, sigma, t_model, noises)
        for fwd_noise in noises:
            zt_src = (1-sigma)*x_src_packed + (sigma)*fwd_noise
            zt_tar = z_edit + zt_src - x_src_packed

            Vt_src = calc_v_flux(pipe,
                                                latents=zt_src,
                                                prompt_embeds=src_prompt_embeds,
                                                pooled_prompt_embeds=src_pooled_prompt_embeds,
                                                guidance=src_guidance,
                                                text_ids=src_text_ids,
                                                latent_image_ids=latent_src_image_ids,
                                                t=t_model)

            Vt_tar = calc_v_flux(pipe,
                                                latents=zt_tar,
                                                prompt_embeds=tar_prompt_embeds,
                                                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                                guidance=tar_guidance,
                                                text_ids=tar_text_ids,
                                                latent_image_ids=latent_tar_image_ids,
                                                t=t_model)
            actual_nfe += 1

            V_delta = flowedit_cfg_like_contrast(Vt_tar, Vt_src)
            zt_src_mid = (zt_src.to(torch.float32) + 0.5 * dt * Vt_src).to(Vt_src.dtype)
            zt_tar_mid = (zt_tar.to(torch.float32) + 0.5 * dt * Vt_tar).to(Vt_tar.dtype)

            Vt_src_mid = calc_v_flux(pipe,
                                                latents=zt_src_mid,
                                                prompt_embeds=src_prompt_embeds,
                                                pooled_prompt_embeds=src_pooled_prompt_embeds,
                                                guidance=src_guidance,
                                                text_ids=src_text_ids,
                                                latent_image_ids=latent_src_image_ids,
                                                t=t_mid_model)

            Vt_tar_mid = calc_v_flux(pipe,
                                                latents=zt_tar_mid,
                                                prompt_embeds=tar_prompt_embeds,
                                                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                                guidance=tar_guidance,
                                                text_ids=tar_text_ids,
                                                latent_image_ids=latent_tar_image_ids,
                                                t=t_mid_model)
            actual_nfe += 1

            V_delta_mid = pc_guidance_weight * flowedit_cfg_like_contrast(Vt_tar_mid, Vt_src_mid)
            if combine_mode == "additive":
                V_hat = V_delta + alpha * V_delta_mid
            elif combine_mode == "interpolate":
                V_hat = (1 - alpha) * V_delta + alpha * V_delta_mid
            elif combine_mode == "cfg_like_interpolate":
                V_hat = (1 - alpha) * V_delta + alpha * V_delta_mid
            elif combine_mode == "cfgpp_no_first_term":
                V_hat = alpha * V_delta_mid
            else:
                raise ValueError(f"Unsupported FlowEdit PC combine_mode: {combine_mode}")
            V_hat_avg += (1/n_avg) * V_hat
        return V_hat_avg

    def flowedit_bridge_pc_delta(z_edit, sigma, t_model, sigma_mid, t_mid_model, dt, noises, correction_mode="interpolate"):
        V_delta = flowedit_delta(z_edit, sigma, t_model, noises)
        alpha = scheduled_flowedit_alpha(sigma, pc_guidance_lambda, pc_guidance_gamma, pc_enable_below_t)
        if float(alpha.max().item()) <= 0:
            return V_delta
        zt_mid = (z_edit.to(torch.float32) + 0.5 * dt * V_delta).to(V_delta.dtype)
        V_delta_mid = flowedit_delta(zt_mid, sigma_mid, t_mid_model, noises)
        if correction_mode == "directional":
            return blend_direction_preserve_norm(V_delta, V_delta_mid, alpha, pc_guidance_weight)
        if correction_mode == "dir_mag":
            return blend_direction_with_magnitude_recovery(
                V_delta,
                V_delta_mid,
                alpha,
                mid_weight=pc_guidance_weight,
                magnitude_weight=pc_magnitude_weight,
                magnitude_min=pc_magnitude_min,
                magnitude_max=pc_magnitude_max,
            )
        return (1 - alpha) * V_delta + alpha * (pc_guidance_weight * V_delta_mid)

    def target_velocity(z_tar, t_model):
        nonlocal actual_nfe
        actual_nfe += 1
        return calc_v_flux(pipe,
                            latents=z_tar,
                            prompt_embeds=tar_prompt_embeds,
                            pooled_prompt_embeds=tar_pooled_prompt_embeds,
                            guidance=tar_guidance,
                            text_ids=tar_text_ids,
                            latent_image_ids=latent_tar_image_ids,
                            t=t_model)

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        if i+1 < len(timesteps):
            t_next = timesteps[i+1]
        else:
            t_next = torch.zeros_like(t).to(t.device)
        dt = t_im1 - t_i
        t_mid = t + 0.5 * (t_next - t)
        sigma_mid = t_i + 0.5 * dt

        if T_steps - i > n_min:

            fwd_noises = [torch.randn_like(x_src_packed).to(x_src_packed.device) for _ in range(n_avg)]
            if solver_type in {
                "flowedit_pc_additive",
                "flowedit_pc_interpolate",
                "flowedit_cfg_like_interpolate",
                "flowedit_cfgpp_no_first_term",
            }:
                V_delta_avg = flowedit_pc_delta(
                    zt_edit,
                    t_i,
                    t,
                    sigma_mid,
                    t_mid,
                    dt,
                    fwd_noises,
                    "cfg_like_interpolate"
                    if solver_type == "flowedit_cfg_like_interpolate"
                    else "cfgpp_no_first_term"
                    if solver_type == "flowedit_cfgpp_no_first_term"
                    else "interpolate"
                    if solver_type == "flowedit_pc_interpolate"
                    else "additive",
                )
            elif solver_type in {"flowedit_bridge_interpolate", "flowedit_bridge_directional", "flowedit_bridge_dir_mag"}:
                V_delta_avg = flowedit_bridge_pc_delta(
                    zt_edit,
                    t_i,
                    t,
                    sigma_mid,
                    t_mid,
                    dt,
                    fwd_noises,
                    correction_mode="directional"
                    if solver_type == "flowedit_bridge_directional"
                    else "dir_mag"
                    if solver_type == "flowedit_bridge_dir_mag"
                    else "interpolate",
                )
            else:
                V_delta_avg = flowedit_delta(zt_edit, t_i, t, fwd_noises)

            if solver_type == "midpoint":
                zt_mid = (zt_edit.to(torch.float32) + 0.5 * dt * V_delta_avg).to(V_delta_avg.dtype)
                V_delta_avg = flowedit_delta(zt_mid, sigma_mid, t_mid, fwd_noises)

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + dt * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = target_velocity(xt_tar, t)

            if solver_type == "midpoint":
                xt_mid = (xt_tar.to(torch.float32) + 0.5 * dt * Vt_tar).to(Vt_tar.dtype)
                Vt_tar = target_velocity(xt_mid, t_mid)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + dt * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    if return_stats:
        return unpacked_out, {"actual_nfe": actual_nfe}
    return unpacked_out
