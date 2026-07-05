"""Оркестратор пайплайна: запускает этапы последовательно.

Каждый этап создает собственный ClearML Task (эксперимент), так что в
веб-интерфейсе ClearML (http://localhost:8080) виден весь пайплайн:
01_prepare_data -> 02_preprocess -> 03_features -> 04_build_dataset -> train.

Примеры:
    python -m src.pipeline                       # весь пайплайн
    python -m src.pipeline --stages train        # только обучение
    python -m src.pipeline --stages preprocess,features,dataset,train
"""

from __future__ import annotations

import argparse
import time

from src import build_dataset, features, prepare_data, preprocess, train
from src.common import load_config, log, setup_logging

STAGES = {
    "prepare": (prepare_data.main, "pipeline"),
    "preprocess": (preprocess.main, "pipeline"),
    "features": (features.main, "pipeline"),
    "dataset": (build_dataset.main, "pipeline"),
    "train": (train.main, "train"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="конфиг data-этапов")
    parser.add_argument("--train-config", default="configs/train.yaml",
                        help="конфиг обучения")
    parser.add_argument(
        "--stages",
        default="all",
        help="какие этапы запускать: all или список через запятую "
        f"из {list(STAGES)}",
    )
    args = parser.parse_args()
    setup_logging()

    stages = list(STAGES) if args.stages == "all" else [
        s.strip() for s in args.stages.split(",")
    ]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"Неизвестные этапы: {unknown}. Доступны: {list(STAGES)}")

    configs = {
        "pipeline": load_config(args.config),
        "train": load_config(args.train_config),
    }

    for stage in stages:
        fn, cfg_name = STAGES[stage]
        log.info("=" * 60)
        log.info(">>> Этап: %s", stage)
        started = time.monotonic()
        fn(configs[cfg_name])
        log.info("<<< Этап %s завершен за %.1f c", stage, time.monotonic() - started)


if __name__ == "__main__":
    main()
