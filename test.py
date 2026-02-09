#!/usr/bin/env python
# coding=utf-8

from pathlib import Path

import hydra
from ovg.benchmarker import BenchmarkRunner
from hydra.core.hydra_config import HydraConfig

from ovg.utils.helpers import omegaconf_select


@hydra.main(version_base="2.5")
def main(cfg):
    run_dir = HydraConfig.get().run.dir
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    cfg.run_dir = omegaconf_select(cfg, "run_dir", default=run_dir)

    benchmark_runner = BenchmarkRunner(cfg)
    benchmark_runner.setup()
    results = benchmark_runner.benchmark()
    benchmark_runner.log_results_wandb(results)
    benchmark_runner.run_logger.close()
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
