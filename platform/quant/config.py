# -*- coding: utf-8 -*-
"""全局路径配置。所有路径可被环境变量覆盖，为上云/迁移预留。"""
import os
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]  # .../astro-blog/platform
REPO_DIR = PLATFORM_DIR.parent  # .../astro-blog

# Parquet 数据湖（不进 git）
DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", str(PLATFORM_DIR / "quant-data")))

# 发布产物目录 = Astro 的 public/data（进 git，随 Vercel 部署）
PUBLISH_DIR = Path(os.environ.get("QUANT_PUBLISH_DIR", str(REPO_DIR / "public" / "data")))

# 每日增量发布的代码清单（进 git，launchd 任务读它）
CODES_FILE = Path(os.environ.get("QUANT_CODES_FILE", str(PLATFORM_DIR / "codes.txt")))

DEFAULT_LOOKBACK_YEARS = 10


def ensure_dirs() -> None:
    (DATA_DIR / "daily_bar").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "adjust_factor").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "meta").mkdir(parents=True, exist_ok=True)
