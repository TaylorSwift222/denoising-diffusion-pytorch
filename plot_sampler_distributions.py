import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

from denoising_diffusion_pytorch.curriculum_sampler import CurriculumSampler


CURRICULUM_MODES = ("low_to_high", "high_to_low", "mid_to_ends")


def load_yaml(path):
    if yaml is None:
        raise ImportError("PyYAML is required to read config files") from YAML_IMPORT_ERROR

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sampler_kwargs(config):
    sampler_cfg = config.get("sampler", {})
    train_cfg = config.get("train", {})
    diffusion_cfg = config.get("diffusion", {})

    return dict(
        num_timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        total_train_steps=int(train_cfg["num_steps"]),
        curriculum_ratio=float(sampler_cfg.get("curriculum_ratio", 0.5)),
        sigma_soft=float(sampler_cfg.get("sigma_soft", 0.08)),
        rho=float(sampler_cfg.get("rho", 0.1)),
        beta_start=float(sampler_cfg.get("beta_start", 1e-4)),
        beta_end=float(sampler_cfg.get("beta_end", 0.02)),
    )


def build_static_sampler(mode, kwargs, exact, integration_steps):
    if exact:
        return CurriculumSampler.create_static_from_curriculum(
            curriculum_mode=mode,
            **kwargs,
        )

    temp = CurriculumSampler(mode=mode, **kwargs)
    marginal = temp.compute_marginal(num_steps=integration_steps)
    return CurriculumSampler(mode="static", static_marginal=marginal, **kwargs)


def step_label(step, curriculum_boundary):
    if step == curriculum_boundary - 1:
        return f"{curriculum_boundary // 1000}k-1"
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def default_steps(total_steps, curriculum_ratio):
    boundary = int(total_steps * curriculum_ratio)
    steps = [0, total_steps // 4, max(boundary - 1, 0), boundary, total_steps]
    return list(dict.fromkeys(steps))


def write_distribution_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "step", "timestep", "z", "probability", "prob_over_uniform"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_curriculum_distributions(samplers, steps, boundary, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(samplers),
        ncols=len(steps),
        figsize=(3.2 * len(steps), 2.6 * len(samplers)),
        sharex=True,
        sharey=True,
    )
    if len(samplers) == 1:
        axes = np.array([axes])

    for row_idx, (mode, sampler) in enumerate(samplers.items()):
        uniform_prob = 1.0 / sampler.T
        for col_idx, step in enumerate(steps):
            ax = axes[row_idx, col_idx]
            probs = sampler.probabilities(step)
            order = np.argsort(sampler.z)
            ax.plot(sampler.z[order], probs[order] / uniform_prob, linewidth=1.8)
            ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title(step_label(step, boundary), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{mode}\nq(t) / uniform")
            if row_idx == len(samplers) - 1:
                ax.set_xlabel("log-SNR rank z")
            ax.grid(True, alpha=0.2)

    fig.suptitle("Curriculum timestep sampling distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_static_marginals(static_samplers, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, sampler in static_samplers.items():
        uniform_prob = 1.0 / sampler.T
        probs = sampler.probabilities(0)
        order = np.argsort(sampler.z)
        ax.plot(sampler.z[order], probs[order] / uniform_prob, label=label, linewidth=2)

    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35, label="uniform")
    ax.set_xlabel("log-SNR rank z (0 = high noise, 1 = low noise)")
    ax.set_ylabel("marginal q(t) / uniform")
    ax.set_title("Matched-static timestep marginals")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot curriculum timestep sampling distributions.")
    parser.add_argument("--config", type=str, default="configs/cifar10_curriculum.yaml")
    parser.add_argument("--output-dir", type=str, default="output/sampler_distributions")
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--static-integration-steps", type=int, default=50000)
    parser.add_argument("--exact-static", action="store_true", help="Use exact S-step marginal for static plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    kwargs = sampler_kwargs(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = kwargs["total_train_steps"]
    curriculum_ratio = kwargs["curriculum_ratio"]
    boundary = int(total_steps * curriculum_ratio)
    steps = args.steps or default_steps(total_steps, curriculum_ratio)

    samplers = {
        mode: CurriculumSampler(mode=mode, **kwargs)
        for mode in CURRICULUM_MODES
    }
    static_samplers = {
        f"static_{mode}": build_static_sampler(
            mode,
            kwargs,
            exact=args.exact_static,
            integration_steps=args.static_integration_steps,
        )
        for mode in CURRICULUM_MODES
    }

    rows = []
    for mode, sampler in samplers.items():
        uniform_prob = 1.0 / sampler.T
        for step in steps:
            probs = sampler.probabilities(step)
            for t, (z, p) in enumerate(zip(sampler.z, probs)):
                rows.append(
                    {
                        "mode": mode,
                        "step": step,
                        "timestep": t,
                        "z": z,
                        "probability": p,
                        "prob_over_uniform": p / uniform_prob,
                    }
                )

    for mode, sampler in static_samplers.items():
        uniform_prob = 1.0 / sampler.T
        probs = sampler.probabilities(0)
        for t, (z, p) in enumerate(zip(sampler.z, probs)):
            rows.append(
                {
                    "mode": mode,
                    "step": "static",
                    "timestep": t,
                    "z": z,
                    "probability": p,
                    "prob_over_uniform": p / uniform_prob,
                }
            )

    write_distribution_csv(rows, output_dir / "sampler_distributions.csv")
    plot_curriculum_distributions(
        samplers,
        steps,
        boundary,
        output_dir / "curriculum_distributions.png",
    )
    plot_static_marginals(
        static_samplers,
        output_dir / "static_marginals.png",
    )

    print(f"Wrote sampler visualizations to {output_dir}")
    print(f"Curriculum boundary step: {boundary} (step >= boundary is uniform)")


if __name__ == "__main__":
    main()
