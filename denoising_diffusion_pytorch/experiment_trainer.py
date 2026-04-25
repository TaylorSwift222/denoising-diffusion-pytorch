import math
from pathlib import Path

import torch
from torchvision import utils
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Trainer, divisible_by, num_to_groups
from denoising_diffusion_pytorch.experiment_utils import save_reverse_trajectory, save_train_triplet


class ExperimentTrainer(Trainer):
    """
    Project-level trainer for experiment logging.

    It keeps the upstream Trainer behavior, while adding optional wandb logging
    and train-step triplet visualization from the real batch/timestep/noise used
    in the current optimization step.
    """

    def __init__(
        self,
        *args,
        wandb_config=None,
        visualization_config=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wandb_config = wandb_config or {}
        self.visualization_config = visualization_config or {}
        self.wandb_run = None
        self.wandb = None

        if self.accelerator.is_main_process:
            self._maybe_init_wandb()

    def _maybe_init_wandb(self):
        mode = self.wandb_config.get("mode", "disabled")
        if mode == "disabled":
            return

        import wandb

        self.wandb = wandb
        self.wandb_run = wandb.init(
            project=self.wandb_config.get("project"),
            name=self.wandb_config.get("run_name"),
            mode=mode,
            config=self.wandb_config.get("config", {}),
            dir=self.wandb_config.get("dir"),
            resume=self.wandb_config.get("resume"),
            id=self.wandb_config.get("id"),
        )

    def _wandb_log(self, data, step=None):
        if self.wandb_run is not None:
            self.wandb.log(data, step=step)

    def _sampler_metadata(self):
        model = self.accelerator.unwrap_model(self.model)
        sampler = getattr(model, "timestep_sampler", None)
        if sampler is None or not hasattr(sampler, "metadata"):
            return {}
        return sampler.metadata(self.step)

    def _should_log_triplet(self):
        cfg = self.visualization_config.get("train_triplet", {})
        if not cfg.get("enabled", False):
            return False
        every = int(cfg.get("every_n_steps", 0))
        return every > 0 and divisible_by(self.step + 1, every)

    def _log_train_triplet(self, vis_payload):
        cfg = self.visualization_config.get("train_triplet", {})
        log_images = int(cfg.get("log_images_per_step", 8))
        path = self.results_folder / "visuals" / f"train-triplet-{self.step}.png"
        save_train_triplet(vis_payload, path, log_images_per_step=log_images)
        if self.wandb_run is not None:
            self._wandb_log({"visual/train_triplet": self.wandb.Image(str(path))}, step=self.step)

    def _maybe_log_reverse_trajectory(self, milestone):
        cfg = self.visualization_config.get("reverse_trajectory", {})
        if not cfg.get("enabled", False):
            return

        path = self.results_folder / "visuals" / f"reverse-trajectory-{milestone}.png"
        save_reverse_trajectory(
            self.ema.ema_model,
            path,
            batch_size=int(cfg.get("batch_size", 4)),
            num_stages=int(cfg.get("num_stages", 8)),
            seed=cfg.get("seed"),
        )
        if self.wandb_run is not None:
            self._wandb_log({"visual/reverse_trajectory": self.wandb.Image(str(path))}, step=self.step)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.
                vis_payload = None

                for micro_step in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    return_vis = self._should_log_triplet() and micro_step == 0

                    with self.accelerator.autocast():
                        loss_out = self.model(data, step=self.step, return_vis=return_vis)
                        if return_vis:
                            loss, vis_payload = loss_out
                        else:
                            loss = loss_out

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()
                    if self.wandb_run is not None:
                        log_data = {"train/loss": total_loss}
                        log_data.update(self._sampler_metadata())
                        self._wandb_log(log_data, step=self.step)

                    if vis_payload is not None:
                        self._log_train_triplet(vis_payload)

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)
                        sample_path = self.results_folder / f"sample-{milestone}.png"
                        utils.save_image(all_images, str(sample_path), nrow=int(math.sqrt(self.num_samples)))
                        if self.wandb_run is not None:
                            self._wandb_log({"sample/grid": self.wandb.Image(str(sample_path))}, step=self.step)
                        self._maybe_log_reverse_trajectory(milestone)

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f"fid_score: {fid_score}")
                            self._wandb_log({"eval/fid": fid_score}, step=self.step)

                        if self.save_best_and_latest_only:
                            if not self.calculate_fid:
                                raise RuntimeError("save_best_and_latest_only=True requires calculate_fid=True")
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print("training complete")
        if self.wandb_run is not None:
            self.wandb.finish()
