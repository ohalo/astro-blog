# -*- coding: utf-8 -*-
"""Parquet 数据湖：按年分区写入、幂等 upsert、DuckDB 统一查询出口。

分区布局（与规划书 5.2 一致的 MVP 版）：
    quant-data/daily_bar/year=YYYY/data.parquet
    quant-data/adjust_factor/year=YYYY/data.parquet
    quant-data/meta/trade_calendar.parquet
    quant-data/meta/instruments.parquet
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .. import config


def _upsert_yearly(df: pd.DataFrame, table_dir: Path, date_col: str = "trade_date") -> dict:
    """通用：按年分区 upsert，(ts_code, trade_date) 去重、新数据为准（keep=last）。"""
    if df.empty:
        return {}
    df = df.copy()
    df[date_col] = df[date_col].astype(str)
    df["_year"] = df[date_col].str[:4]
    counts: dict[str, int] = {}
    for year, part in df.groupby("_year"):
        part = part.drop(columns=["_year"])
        pdir = table_dir / f"year={year}"
        pdir.mkdir(parents=True, exist_ok=True)
        path = pdir / "data.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = (part.sort_values(["ts_code", date_col])
                    .drop_duplicates(["ts_code", date_col], keep="last")
                    .reset_index(drop=True))
        part.to_parquet(path, index=False, compression="zstd")
        counts[str(year)] = len(part)
    return counts


def upsert_daily_bar(df: pd.DataFrame) -> dict:
    return _upsert_yearly(df, config.DATA_DIR / "daily_bar")


def upsert_adjust_factor(df: pd.DataFrame) -> dict:
    return _upsert_yearly(df, config.DATA_DIR / "adjust_factor")


def write_meta(name: str, df: pd.DataFrame) -> Path:
    path = config.DATA_DIR / "meta" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


# ---------------- 查询出口（全部走 DuckDB，SQL 里用 glob 路径） ----------------

def _sql_path(rel_glob: str) -> str:
    return str((config.DATA_DIR / rel_glob).as_posix())


def query(sql: str, params=None) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        res = con.execute(sql, params or [])
        # duckdb>=1.4(nanobind) 起 fetch_df/fetchdf 为方法，.df 属性已不可当 DataFrame 用
        if hasattr(res, "fetch_df"):
            return res.fetch_df()
        return res.fetchdf()
    finally:
        con.close()


def daily_bar_sql() -> str:
    return f"read_parquet('{_sql_path('daily_bar/*/*.parquet')}', hive_partitioning=0)"


def lake_codes() -> list[str]:
    import glob
    # DuckDB 在完全无分区目录时 glob 会报错，先用文件系统探一次
    if not glob.glob(str(config.DATA_DIR / "daily_bar" / "*" / "*.parquet")):
        return []
    df = query(f"SELECT DISTINCT ts_code FROM {daily_bar_sql()} ORDER BY ts_code")
    return df["ts_code"].tolist()


def latest_trade_date(ts_code: str) -> str | None:
    df = query(
        f"SELECT max(trade_date) AS d FROM {daily_bar_sql()} WHERE ts_code = ?",
        [ts_code],
    )
    d = df.iloc[0, 0]
    return None if d is None or pd.isna(d) else str(d)


def load_daily_bar(ts_code: str, count: int | None = None) -> pd.DataFrame:
    """载入某只票全部日线（按日期升序）。停牌行（close NaN）由调用方决定去留。"""
    sql = (f"SELECT * FROM {daily_bar_sql()} WHERE ts_code = ? ORDER BY trade_date")
    df = query(sql, [ts_code])
    if count is not None and len(df) > count:
        df = df.iloc[-count:]
    return df.reset_index(drop=True)


def meta_df(name: str) -> pd.DataFrame:
    path = config.DATA_DIR / "meta" / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
