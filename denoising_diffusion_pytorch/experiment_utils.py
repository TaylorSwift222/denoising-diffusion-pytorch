import csv
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torchvision import utils

try:
    import yaml
except ImportError as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

from denoising_diffusion_pytorch.curriculum_sampler import CurriculumSampler
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, num_to_groups


def load_yaml(path):
    if yaml is None:
        raise ImportError("PyYAML is required to read experiment configs") from YAML_IMPORT_ERROR

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deep_update(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_tuple(value):
    if value is None:
        return None
    return tuple(value) if isinstance(value, list) else value


def build_sampler(config):
    sampler_cfg = config.get("sampler", {})
    train_cfg = config.get("train", {})
    diffusion_cfg = config.get("diffusion", {})

    mode = sampler_cfg.get("mode", "uniform")
    total_train_steps = int(train_cfg["num_steps"])
    num_timesteps = int(diffusion_cfg.get("timesteps", 1000))

    kwargs = dict(
        num_timesteps=num_timesteps,
        total_train_steps=total_train_steps,
        curriculum_ratio=float(sampler_cfg.get("curriculum_ratio", 0.5)),
        sigma_soft=float(sampler_cfg.get("sigma_soft", 0.08)),
        rho=float(sampler_cfg.get("rho", 0.1)),
        beta_start=float(sampler_cfg.get("beta_start", 1e-4)),
        beta_end=float(sampler_cfg.get("beta_end", 0.02)),
    )

    if mode in ("none", "uniform"):
        return None

    static_prefix = "static_"
    if mode.startswith(static_prefix):
        curriculum_mode = mode[len(static_prefix):]
        print(f"Computing exact matched-static marginal for {curriculum_mode}; this runs once at startup.")
        return CurriculumSampler.create_static_from_curriculum(
            curriculum_mode=curriculum_mode,
            **kwargs,
        )

    return CurriculumSampler(mode=mode, **kwargs)


def build_diffusion(config, timestep_sampler=None):
    model_cfg = config.get("model", {})
    diffusion_cfg = config.get("diffusion", {})

    model = Unet(
        dim=int(model_cfg.get("dim", 64)),
        init_dim=model_cfg.get("init_dim"),
        out_dim=model_cfg.get("out_dim"),
        dim_mults=_as_tuple(model_cfg.get("dim_mults", (1, 2, 4, 8))),
        channels=int(model_cfg.get("channels", 3)),
        self_condition=bool(model_cfg.get("self_condition", False)),
        learned_variance=bool(model_cfg.get("learned_variance", False)),
        learned_sinusoidal_cond=bool(model_cfg.get("learned_sinusoidal_cond", False)),
        random_fourier_features=bool(model_cfg.get("random_fourier_features", False)),
        learned_sinusoidal_dim=int(model_cfg.get("learned_sinusoidal_dim", 16)),
        sinusoidal_pos_emb_theta=int(model_cfg.get("sinusoidal_pos_emb_theta", 10000)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attn_dim_head=_as_tuple(model_cfg.get("attn_dim_head", 32)),
        attn_heads=_as_tuple(model_cfg.get("attn_heads", 4)),
        full_attn=_as_tuple(model_cfg.get("full_attn")),
        flash_attn=bool(model_cfg.get("flash_attn", False)),
    )

    return GaussianDiffusion(
        model,
        image_size=diffusion_cfg.get("image_size", 32),
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        sampling_timesteps=diffusion_cfg.get("sampling_timesteps", None),
        objective=diffusion_cfg.get("objective", "pred_noise"),
        beta_schedule=diffusion_cfg.get("beta_schedule", "linear"),
        schedule_fn_kwargs=diffusion_cfg.get("schedule_fn_kwargs", {}),
        ddim_sampling_eta=float(diffusion_cfg.get("ddim_sampling_eta", 0.0)),
        auto_normalize=bool(diffusion_cfg.get("auto_normalize", True)),
        offset_noise_strength=float(diffusion_cfg.get("offset_noise_strength", 0.0)),
        min_snr_loss_weight=bool(diffusion_cfg.get("min_snr_loss_weight", False)),
        min_snr_gamma=float(diffusion_cfg.get("min_snr_gamma", 5.0)),
        immiscible=bool(diffusion_cfg.get("immiscible", False)),
        timestep_sampler=timestep_sampler,
    )


def build_experiment(config):
    sampler = build_sampler(config)
    diffusion = build_diffusion(config, timestep_sampler=sampler)
    return diffusion, sampler


def triplet_grid_from_vis(vis, log_images_per_step=8):
    n = min(int(log_images_per_step), vis["x_start"].shape[0])
    triplets = torch.stack(
        (
            vis["x_t"][:n].detach().cpu(),
            vis["pred_x_start"][:n].detach().cpu(),
            vis["x_start"][:n].detach().cpu(),
        ),
        dim=1,
    )
    triplets = triplets.reshape(n * 3, *triplets.shape[2:])
    return utils.make_grid(triplets, nrow=3)


def save_train_triplet(vis, path, log_images_per_step=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = triplet_grid_from_vis(vis, log_images_per_step=log_images_per_step)
    utils.save_image(grid, str(path))
    return grid


@torch.inference_mode()
def save_reverse_trajectory(model, path, batch_size=4, num_stages=8, seed=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if seed is None:
        samples = model.sample(batch_size=batch_size, return_all_timesteps=True).detach().cpu()
    else:
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        samples = model.sample(batch_size=batch_size, return_all_timesteps=True).detach().cpu()
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    _, steps, *_ = samples.shape
    stage_indices = torch.linspace(0, steps - 1, steps=num_stages).round().long()
    selected = samples[:, stage_indices]
    grid = selected.reshape(batch_size * num_stages, *selected.shape[2:])
    grid = utils.make_grid(grid, nrow=num_stages)
    utils.save_image(grid, str(path))
    return grid


def save_sample_grid(images, path, nrow=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if nrow is None:
        nrow = int(math.sqrt(images.shape[0]))
    utils.save_image(images, str(path), nrow=nrow)


def save_sampler_plot(sampler, path, step=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    probs = sampler.probabilities(step)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(probs)), probs)
    ax.set_xlabel("timestep")
    ax.set_ylabel("probability")
    ax.set_title(f"{sampler.mode} at step {step}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def append_metrics_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
