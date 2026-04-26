import argparse
import datetime
import json
import sys
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TeeStream:
    """Duplicate writes to both the original stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup_eval_logging(run_dir):
    """Create a timestamped eval log file and tee stdout/stderr."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(run_dir)
    log_path = run_dir / f"eval_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path, log_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument("--num-is-samples", type=int, default=None)
    parser.add_argument("--is-splits", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compute-is", action="store_true")
    parser.add_argument("--skip-fid", action="store_true", help="Only compute preview/IS metrics; useful for adding IS after FID is done.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip sample grid and reverse trajectory generation.")
    parser.add_argument("--metrics-file", type=str, default="metrics.csv", help="Metrics CSV name or path. Relative paths are placed under --run-dir.")
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

    if args.skip_fid and not (args.compute_is or eval_cfg.get("compute_is", False)):
        raise ValueError("--skip-fid was set, but IS is disabled; nothing to evaluate.")

    if not args.skip_visuals:
        sample_count = int(eval_cfg.get("num_preview_samples", 25))
        with torch.inference_mode():
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

    fid = None
    if not args.skip_fid:
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
        num_is_samples = int(args.num_is_samples or eval_cfg.get("num_is_samples", num_fid_samples))
        is_splits = int(args.is_splits or eval_cfg.get("is_splits", 10))
        is_mean, is_std = inception_score(
            model,
            batch_size=batch_size,
            num_samples=num_is_samples,
            device=device,
            splits=is_splits,
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
    metrics_path = Path(args.metrics_file)
    if not metrics_path.is_absolute():
        metrics_path = run_dir / metrics_path
    append_metrics_csv(metrics_path, row)
    print(row)


def main():
    args = parse_args()
    config = load_config(args)
    milestones = args.milestones if args.milestones is not None else [args.milestone]

    # Setup logging
    run_dir = Path(args.run_dir) if args.run_dir else Path(".")
    log_path, log_file = setup_eval_logging(run_dir)
    try:
        sep = "=" * 60
        print(sep)
        print(f"Eval started : {datetime.datetime.now().isoformat()}")
        print(f"Run dir      : {run_dir}")
        print(f"Milestones   : {milestones}")
        print(f"Log file     : {log_path}")
        print(sep)

        for milestone in milestones:
            evaluate_one(config, args, milestone)

        print(f"\n{sep}")
        print(f"Eval complete: {datetime.datetime.now().isoformat()}")
        print(sep)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
