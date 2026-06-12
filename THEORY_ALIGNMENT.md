# FlowEdit Theory Alignment

This note aligns the theory variants from the papers and PPT with the actual solver implementations in this repo.

## 1. Original FlowEdit

Base editing field:

```text
G_t = v_t^tar - v_t^src
```

In code:
- Source/target bridge construction is in [FlowEdit_utils.py](FlowEdit_utils.py) for SD3 and FLUX.
- The Euler baseline corresponds to `solver_type: "euler"`.

## 2. Branch-Aware Midpoint Formula From The PPT

This is the original branch-wise midpoint idea:

```text
G_mid^branch = \tilde v_{t+h/2}^tar - \tilde v_{t+h/2}^src
hat G_t = (1 - alpha(t)) G_t + alpha(t) G_mid^branch
alpha(t) = lambda (1 - t)^gamma
```

Interpretation:
- Take half-steps on the source branch and target branch separately.
- Recompute the branch velocities at the midpoint.
- Interpolate toward the midpoint contrast instead of adding a second guidance force.

In code:
- Branch midpoint predictor-corrector is implemented by `flowedit_pc_delta(...)` in [FlowEdit_utils.py](FlowEdit_utils.py).
- The exact branch half-step and interpolation rule are implemented in [FlowEdit_utils.py](FlowEdit_utils.py).
- This theory variant maps to `solver_type: "flowedit_cfg_like_interpolate"`.

## 3. Reverse Derivation From "Remove The CFG++ First Term"

The PPT argument was:

```text
hat G_t = alpha(t) G_mid
```

under-edits because it removes the base editing anchor.

So the missing anchor should be restored with FlowEdit's own contrastive field:

```text
G_t = v_t^tar - v_t^src
hat G_t = (1 - alpha(t)) G_t + alpha(t) G_mid
```

This derivation is conceptually aligned with `flowedit_cfg_like_interpolate`.

## 4. Bridge-Consistent Deep Adaptation

The branch-wise midpoint is faithful to the PPT, but it advances source and target branches independently before subtraction.
The deeper adaptation used here keeps FlowEdit's shared-noise bridge exact at both the current step and the midpoint:

```text
G_t = FlowEditDelta(z_edit, t)
z_mid = z_edit + (h / 2) G_t
G_mid^bridge = FlowEditDelta(z_mid, t + h / 2)
hat G_t = (1 - alpha(t)) G_t + alpha(t) G_mid^bridge
```

Interpretation:
- Midpoint is taken in edit-latent space.
- Each midpoint evaluation reconstructs the same FlowEdit bridge again, rather than midpointing the two branches separately.

In code:
- Bridge midpoint correction is implemented by `flowedit_bridge_pc_delta(...)` in [FlowEdit_utils.py](FlowEdit_utils.py) for SD3 and FLUX.
- This theory variant maps to `solver_type: "flowedit_bridge_interpolate"`.

## 5. Direction-Preserving Deep Adaptation

To reduce over-editing, the midpoint term can be used only as a direction corrector while preserving the norm of the base FlowEdit field:

```text
dir(hat G_t) = blend(dir G_t, dir G_mid^bridge)
||hat G_t|| = ||G_t||
```

In code:
- Direction-preserving blend is implemented in [FlowEdit_utils.py](FlowEdit_utils.py).
- This theory variant maps to `solver_type: "flowedit_bridge_directional"`.

## 6. Late-Stage Correction Schedule

The deep adaptation also supports activating midpoint correction only in later denoising stages:

```text
alpha(t) = lambda (1 - t)^gamma, only when t <= enable_below_t
```

In code:
- The schedule is implemented in [FlowEdit_utils.py](FlowEdit_utils.py).
- The corresponding runtime parameter is `pc_enable_below_t`.

## 7. Recommended Evaluation Protocol

Use [SD3_theory_alignment_same_nfe.yaml](SD3_theory_alignment_same_nfe.yaml) to compare:
- Original FlowEdit Euler
- PPT-faithful branch-aware interpolation
- Bridge-consistent interpolation
- Direction-preserving bridge interpolation

Then use [SD3_bridge_directional_late_stage.yaml](SD3_bridge_directional_late_stage.yaml) to push quality after the theory alignment pass.

## 8. Runtime Metadata

`run_script.py` now writes theory labels into `outputs/run_summary.csv` and each `prompts.txt`:
- `normalized_solver_type`
- `theory_family`
- `theory_formula`
- `midpoint_space`
- `correction_mode`
- `estimated_nfe`
- `actual_nfe`

This makes cloud experiments directly traceable back to the intended theory variant.
