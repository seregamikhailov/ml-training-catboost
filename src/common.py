"""Общие утилиты пайплайна: конфиги, ClearML, выбор torch-устройства."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("pipeline")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Относительные пути в конфигах считаем от корня проекта."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def init_clearml(cfg: dict, task_name: str, task_type: str = "data_processing"):
    """Создает ClearML Task для этапа пайплайна.

    Возвращает None, если ClearML выключен в конфиге или сервер недоступен —
    пайплайн должен работать и без трекинга.
    """
    clearml_cfg = cfg.get("clearml", {})
    if not clearml_cfg.get("enabled", False):
        log.info("ClearML выключен в конфиге, этап '%s' идет без трекинга", task_name)
        return None
    try:
        from clearml import Task

        task = Task.init(
            project_name=clearml_cfg.get("project", "ShortVideo-Ranking"),
            task_name=task_name,
            task_type=task_type,
            reuse_last_task_id=False,
            auto_connect_frameworks=True,
        )
        task.connect_configuration(cfg, name="config")
        log.info("ClearML task '%s' создан: %s", task_name, task.id)
        return task
    except Exception as e:  # noqa: BLE001 - трекинг не должен ронять пайплайн
        log.warning("Не удалось создать ClearML task '%s': %s", task_name, e)
        return None


def close_clearml(task) -> None:
    if task is not None:
        task.close()


def get_torch_device() -> "str":
    """cuda -> mps -> cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
