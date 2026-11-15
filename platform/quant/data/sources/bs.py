# -*- coding: utf-8 -*-
"""baostock 数据源适配层。

纪律：所有 baostock 调用集中于此，上游接口变动只影响本文件；上层只见 DataFrame。
baostock 返回全是字符串表，本层负责转数值、统一列名。
ts_code 约定沿用 baostock 风格：sh.600519 / sz.000001 / bj.430047。
"""
from __future__ import annotations

import contextlib
import re
from datetime import date

import baostock as bs
import pandas as pd

DAILY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST"
_DAILY_NUMERIC = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pct_chg"]


def normalize_code(code: str) -> str:
    """600519 -> sh.600519；000001 -> sz.000001；430047 -> bj.430047（已带前缀则原样返回）。

    注意：B 股（sh.900xxx / sz.200xxx）会被归到 bj 或直接报错，MVP 阶段不覆盖 B 股。
    """
    c = code.strip().lower()
    if re.fullmatch(r"(sh|sz|bj)\.\d{6}", c):
        return c
    m = re.fullmatch(r"\d{6}", c)
    if not m:
        raise ValueError(f"无法识别的股票代码: {code!r}")
    num = m.group(0)
    if num[0] == "6":  # 60 沪主板 / 68 科创板
        return "sh." + num
    if num[0] in ("0", "3"):  # 00 深主板 / 30 创业板
        return "sz." + num
    if num[0] in ("4", "8", "9"):  # 北交所 43/83/87/920（9 开头 B 股除外）
        if num.startswith("900"):
            raise ValueError(f"MVP 不支持 B 股: {code!r}")
        return "bj." + num
    raise ValueError(f"无法识别市场的代码: {code!r}")


def today_str() -> str:
    return date.today().isoformat()


@contextlib.contextmanager
def session():
    """with session(): ...  包住所有 baostock 查询。"""
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_code} {lg.error_msg}")
    try:
        yield
    finally:
        bs.logout()


def _rs_to_df(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 查询失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def fetch_daily(ts_code: str, start: str, end: str, adjust: str = "none") -> pd.DataFrame:
    """日线（默认未复权）。返回列: ts_code, trade_date(str), open..close, preclose,
    volume(股), amount(元), turn(%), pct_chg(%), tradestatus(1正常/0停牌), is_st。
    """
    flag = {"none": "3", "front": "2", "back": "1"}[adjust]
    with session():
        rs = bs.query_history_k_data_plus(
            normalize_code(ts_code), DAILY_FIELDS,
            start_date=start, end_date=end, frequency="d", adjustflag=flag,
        )
        df = _rs_to_df(rs)
    if df.empty:
        return df
    df = df.rename(columns={"date": "trade_date", "code": "ts_code", "pctChg": "pct_chg", "isST": "is_st"})
    for col in _DAILY_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # 停牌日空串 -> NaN
    for col in ("tradestatus", "is_st"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def fetch_adjust_factor(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """复权因子表（除息日粒度）。实测真实返回字段（2026-11）：
    code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor。
    统一成 ts_code, trade_date(=除息日), fore/back/adjust_factor 后交给引擎层算复权口径。"""
    with session():
        rs = bs.query_adjust_factor(code=normalize_code(ts_code), start_date=start, end_date=end)
        df = _rs_to_df(rs)
    if df.empty:
        return df
    df = df.rename(columns={"code": "ts_code", "dividOperateDate": "trade_date",
                            "adjustFactor": "adjust_factor",
                            "foreAdjustFactor": "fore_adjust_factor",
                            "backAdjustFactor": "back_adjust_factor"})
    for col in df.columns:
        if col not in ("ts_code", "trade_date"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def fetch_trade_dates(start: str, end: str) -> pd.DataFrame:
    """交易日历：calendar_date(str), is_trading_day(0/1)。"""
    with session():
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        df = _rs_to_df(rs)
    df["is_trading_day"] = pd.to_numeric(df["is_trading_day"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def fetch_stock_list() -> pd.DataFrame:
    """证券基础信息全表，过滤出股票(type==1)。列: ts_code, name, ipo_date, out_date, status。"""
    with session():
        rs = bs.query_stock_basic()
        df = _rs_to_df(rs)
    if df.empty:
        return df
    df = df.rename(columns={"code": "ts_code", "code_name": "name", "ipoDate": "ipo_date",
                            "outDate": "out_date", "type": "type", "status": "status"})
    df = df[df["type"] == "1"].copy()
    drop_cols = [c for c in ("type", "status") if c in df.columns]
    return df.drop(columns=drop_cols).reset_index(drop=True)


def fetch_name(ts_code: str) -> str | None:
    """单只股票名称（走 query_stock_basic 带 code，轻量）。"""
    with session():
        rs = bs.query_stock_basic(code=normalize_code(ts_code))
        df = _rs_to_df(rs)
    if df.empty:
        return None
    return str(df.iloc[0].get("code_name") or "") or None
