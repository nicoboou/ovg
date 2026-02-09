#!/usr/bin/env python
# coding=utf-8

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="2.5")
def main(cfg: DictConfig) -> None:
    from ovg.utils.config import setup_config
    from ovg.trainer import Trainer

    cfg = setup_config(cfg)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
