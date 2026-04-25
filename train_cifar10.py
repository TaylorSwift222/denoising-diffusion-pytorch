import argparse
from pathlib import Path

from denoising_diffusion_pytorch.experiment_trainer import ExperimentTrainer
from denoising_diffusion_pytorch.experiment_utils import (
    build_experiment,
    deep_update,
    ensure_dir,
    load_yaml,
    save_json,
    save_sampler_plot,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 DDPM curriculum experiments.")
    parser.add_argument("--config", type=str, default="configs/cifar10_curriculum.yaml")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        choices=[
            "uniform",
            "low_to_high",
            "high_to_low",
            "mid_to_ends",
            "static_low_to_high",
            "static_high_to_low",
            "static_mid_to_ends",
        ],
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint milestone to load, e.g. latest, best, 10.")
    return parser.parse_args()


def apply_cli_overrides(config, args):
    overrides = {}

    if args.exp is not None:
        overrides.setdefault("sampler", {})["mode"] = args.exp
        overrides.setdefault("experiment", {})["name"] = args.exp

    if args.name is not None:
        overrides.setdefault("experiment", {})["name"] = args.name

    if args.data_path is not None:
        overrides.setdefault("data", {})["path"] = args.data_path

    if args.output_root is not None:
        overrides.setdefault("output", {})["root"] = args.output_root

    if args.num_steps is not None:
        overrides.setdefault("train", {})["num_steps"] = args.num_steps

    if args.wandb_mode is not None:
        overrides.setdefault("wandb", {})["mode"] = args.wandb_mode

    return deep_update(config, overrides)


def main():
    args = parse_args()
    config = apply_cli_overrides(load_yaml(args.config), args)

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    run_name = config.get("experiment", {}).get("name", config.get("sampler", {}).get("mode", "run"))
    output_root = Path(config.get("output", {}).get("root", "output"))
    results_folder = ensure_dir(output_root / run_name)

    save_json(config, results_folder / "config.json")

    diffusion, sampler = build_experiment(config)
    if sampler is not None:
        save_sampler_plot(sampler, results_folder / "sampler-step0.png", step=0)

    train_cfg = config.get("train", {})
    wandb_cfg = config.get("wandb", {})
    wandb_cfg = {
        **wandb_cfg,
        "run_name": wandb_cfg.get("run_name", run_name),
        "config": config,
        "dir": str(results_folder),
    }

    trainer = ExperimentTrainer(
        diffusion,
        config.get("data", {}).get("path", "data/cifar10_images"),
        train_batch_size=int(train_cfg.get("batch_size", 32)),
        gradient_accumulate_every=int(train_cfg.get("gradient_accumulate_every", 1)),
        augment_horizontal_flip=bool(train_cfg.get("augment_horizontal_flip", True)),
        train_lr=float(train_cfg.get("lr", 1e-4)),
        train_num_steps=int(train_cfg.get("num_steps", 100000)),
        ema_update_every=int(train_cfg.get("ema_update_every", 10)),
        ema_decay=float(train_cfg.get("ema_decay", 0.995)),
        adam_betas=tuple(train_cfg.get("adam_betas", (0.9, 0.99))),
        save_and_sample_every=int(train_cfg.get("save_and_sample_every", 1000)),
        num_samples=int(train_cfg.get("num_samples", 25)),
        results_folder=str(results_folder),
        amp=bool(train_cfg.get("amp", False)),
        mixed_precision_type=train_cfg.get("mixed_precision_type", "fp16"),
        split_batches=bool(train_cfg.get("split_batches", True)),
        convert_image_to=train_cfg.get("convert_image_to"),
        calculate_fid=bool(train_cfg.get("calculate_fid", False)),
        inception_block_idx=int(train_cfg.get("inception_block_idx", 2048)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        num_fid_samples=int(train_cfg.get("num_fid_samples", 50000)),
        save_best_and_latest_only=bool(train_cfg.get("save_best_and_latest_only", False)),
        wandb_config=wandb_cfg,
        visualization_config=config.get("visualization", {}),
    )

    if args.resume is not None:
        trainer.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
