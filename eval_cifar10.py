import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset, cycle
from denoising_diffusion_pytorch.experiment_utils import (
    append_metrics_csv,
    build_diffusion,
    ensure_dir,
    load_yaml,
    num_to_groups,
    save_reverse_trajectory,
)
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 DDPM curriculum checkpoints.")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory containing config.json and checkpoints.")
    parser.add_argument("--config", type=str, default=None, help="YAML config, used when --run-dir is not given.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Overrides --milestone.")
    parser.add_argument("--milestone", type=str, default="latest", help="best, latest, or numeric milestone.")
    parser.add_argument("--milestones", nargs="+", default=None, help="Evaluate multiple milestones from --run-dir.")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--num-fid-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compute-is", action="store_true")
    return parser.parse_args()


def load_config(args):
    if args.run_dir is not None:
        with open(Path(args.run_dir) / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    if args.config is None:
        raise ValueError("Provide either --run-dir or --config")

    return load_yaml(args.config)


def resolve_checkpoint(args, milestone):
    if args.checkpoint is not None:
        return Path(args.checkpoint)

    if args.run_dir is None:
        raise ValueError("--checkpoint is required when --run-dir is not given")

    return Path(args.run_dir) / f"model-{milestone}.pt"


@torch.inference_mode()
def inception_score(model, batch_size, num_samples, device, splits=10):
    from torchvision.models import Inception_V3_Weights, inception_v3

    net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    net.eval()

    preds = []
    for batch in tqdm(num_to_groups(num_samples, batch_size), desc="inception score"):
        samples = model.sample(batch_size=batch)
        samples = F.interpolate(samples, size=(299, 299), mode="bilinear", align_corners=False)
        logits = net(samples)
        preds.append(torch.softmax(logits, dim=1).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    split_scores = []
    for part in torch.tensor_split(torch.from_numpy(preds), splits):
        part = part.clamp_min(1e-12)
        py = part.mean(dim=0, keepdim=True)
        kl = part * (part.log() - py.log())
        split_scores.append(torch.exp(kl.sum(dim=1).mean()).item())

    return float(torch.tensor(split_scores).mean()), float(torch.tensor(split_scores).std())


def evaluate_one(config, args, milestone):
    run_dir = Path(args.run_dir) if args.run_dir is not None else Path(config.get("output", {}).get("root", "output"))
    eval_dir = ensure_dir(run_dir / "eval" / milestone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = build_diffusion(config, timestep_sampler=None).to(device)

    checkpoint_path = resolve_checkpoint(args, milestone)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint["model"])

    model = diffusion
    if args.use_ema and "ema" in checkpoint:
        ema = EMA(diffusion)
        ema.load_state_dict(checkpoint["ema"])
        model = ema.ema_model.to(device)

    model.eval()

    eval_cfg = config.get("eval", {})
    batch_size = int(args.batch_size or eval_cfg.get("batch_size", config.get("train", {}).get("batch_size", 32)))
    num_fid_samples = int(args.num_fid_samples or eval_cfg.get("num_fid_samples", 10000))

    with torch.inference_mode():
        sample_count = int(eval_cfg.get("num_preview_samples", 25))
        samples = torch.cat([model.sample(batch_size=n) for n in num_to_groups(sample_count, batch_size)], dim=0)
        utils.save_image(samples, str(eval_dir / "sample-grid.png"), nrow=int(sample_count ** 0.5))

    reverse_cfg = config.get("visualization", {}).get("reverse_trajectory", {})
    if reverse_cfg.get("enabled", True):
        save_reverse_trajectory(
            model,
            eval_dir / "reverse-trajectory.png",
            batch_size=int(reverse_cfg.get("batch_size", 4)),
            num_stages=int(reverse_cfg.get("num_stages", 8)),
            seed=reverse_cfg.get("seed"),
        )

    ds = Dataset(
        config.get("data", {}).get("path", "data/cifar10_images"),
        diffusion.image_size,
        augment_horizontal_flip=False,
        convert_image_to=config.get("train", {}).get("convert_image_to"),
    )
    dl = cycle(DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True))

    fid_scorer = FIDEvaluation(
        batch_size=batch_size,
        dl=dl,
        sampler=model,
        channels=diffusion.channels,
        accelerator=None,
        stats_dir=str(run_dir),
        device=device,
        num_fid_samples=num_fid_samples,
        inception_block_idx=int(eval_cfg.get("inception_block_idx", 2048)),
    )
    fid = fid_scorer.fid_score()

    is_mean = None
    is_std = None
    if args.compute_is or eval_cfg.get("compute_is", False):
        is_mean, is_std = inception_score(
            model,
            batch_size=batch_size,
            num_samples=int(eval_cfg.get("num_is_samples", num_fid_samples)),
            device=device,
            splits=int(eval_cfg.get("is_splits", 10)),
        )

    row = {
        "run_dir": str(run_dir),
        "milestone": milestone,
        "checkpoint": str(checkpoint_path),
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "num_fid_samples": num_fid_samples,
    }
    append_metrics_csv(run_dir / "metrics.csv", row)
    print(row)


def main():
    args = parse_args()
    config = load_config(args)
    milestones = args.milestones if args.milestones is not None else [args.milestone]

    for milestone in milestones:
        evaluate_one(config, args, milestone)


if __name__ == "__main__":
    main()
