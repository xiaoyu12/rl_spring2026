"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None
    # Whether to torch.compile the training step (faster, but slower startup).
    compile: bool = True


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)
    # Logger fixes its CSV columns from the first logged row, so declare all
    # scalar columns (train + eval) up front so eval rewards land in log.csv too.
    logger.header = [
        "train/loss",
        "train/epoch",
        "train/steps_per_sec",
        "eval/mean_reward",
        "step",
    ]
    with logger.csv_path.open("w") as f:
        f.write(",".join(logger.header) + "\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    def train_step(
        state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(state, action_chunk)
        loss.backward()
        optimizer.step()
        return loss.detach()

    if config.compile:
        train_step = torch.compile(train_step)

    def run_eval(step: int) -> None:
        evaluate_policy(
            model,
            normalizer,
            device,
            chunk_size=config.chunk_size,
            video_size=config.video_size,
            num_video_episodes=config.num_video_episodes,
            flow_num_steps=config.flow_num_steps,
            step=step,
            logger=logger,
        )
        model.train()

    step = 0
    last_eval_step = -1
    loss_sum = 0.0
    loss_count = 0
    t_last = time.perf_counter()
    model.train()
    for epoch in range(config.num_epochs):
        for state, action_chunk in loader:
            state = state.to(device, non_blocking=True)
            action_chunk = action_chunk.to(device, non_blocking=True)

            loss = train_step(state, action_chunk)
            step += 1
            loss_sum += loss.item()
            loss_count += 1

            if step % config.log_interval == 0:
                now = time.perf_counter()
                logger.log(
                    {
                        "train/loss": loss_sum / loss_count,
                        "train/epoch": epoch,
                        "train/steps_per_sec": loss_count / (now - t_last),
                    },
                    step=step,
                )
                loss_sum = 0.0
                loss_count = 0
                t_last = now

            if step % config.eval_interval == 0:
                run_eval(step)
                last_eval_step = step
                t_last = time.perf_counter()

        print(f"epoch {epoch + 1}/{config.num_epochs} | step {step} | loss {loss.item():.4f}")

    # Final evaluation at the end of training (if not just evaluated).
    if last_eval_step != step:
        run_eval(step)

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
