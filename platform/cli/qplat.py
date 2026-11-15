# -*- coding: utf-8 -*-
"""qplat —— 阶段零命令行入口。

用法（platform/ 目录下）：
    .venv/bin/python -m cli.qplat ingest-daily  --code 600519 [--start 2016-11-14 --end 2026-11-14]
    .venv/bin/python -m cli.qplat ingest-adjust --code 600519
    .venv/bin/python -m cli.qplat ingest-calendar
    .venv/bin/python -m cli.qplat ingest-list
    .venv/bin/python -m cli.qplat publish-kline --code 600519 [--count 250]
    .venv/bin/python -m cli.qplat publish-all   [--count 250]
    .venv/bin/python -m cli.qplat daily          # launchd 入口：codes.txt 增量入库+发布+K线
    .venv/bin/python -m cli.qplat status
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta

from quant import config
from quant.data import lake
from quant.data.sources import bs


def _digits(ts_code: str) -> str:
    return ts_code.split(".")[1]


def _default_start() -> str:
    return (date.today() - timedelta(days=365 * config.DEFAULT_LOOKBACK_YEARS + 2)).isoformat()


# ---------------- ingest ----------------

def cmd_ingest_daily(args) -> None:
    ts = bs.normalize_code(args.code)
    start, end = args.start or _default_start(), args.end or bs.today_str()
    df = bs.fetch_daily(ts, start, end, adjust="none")
    if df.empty:
        print(f"[ingest-daily] {ts}: {start}~{end} 无数据")
        return
    counts = lake.upsert_daily_bar(df)
    total = sum(counts.values())
    print(f"[ingest-daily] {ts} {start}~{end} 写入 {total} 行, 分区 {counts}")


def cmd_ingest_adjust(args) -> None:
    ts = bs.normalize_code(args.code)
    start, end = args.start or _default_start(), args.end or bs.today_str()
    df = bs.fetch_adjust_factor(ts, start, end)
    counts = lake.upsert_adjust_factor(df)
    print(f"[ingest-adjust] {ts} 写入 {sum(counts.values())} 行除息因子, 分区 {counts}")


def cmd_ingest_calendar(args) -> None:
    start = args.start or "2010-01-01"
    end = args.end or (date.today() + timedelta(days=400)).isoformat()
    df = bs.fetch_trade_dates(start, end)
    path = lake.write_meta("trade_calendar", df)
    n = int((df["is_trading_day"] == 1).sum())
    print(f"[ingest-calendar] {start}~{end} 交易日 {n} 天 -> {path.name}")


def cmd_ingest_list(args) -> None:
    df = bs.fetch_stock_list()
    path = lake.write_meta("instruments", df)
    print(f"[ingest-list] 股票 {len(df)} 只 -> {path.name}")


# ---------------- publish ----------------

def _name_of(ts_code: str) -> str | None:
    inst = lake.meta_df("instruments")
    if not inst.empty:
        hit = inst[inst["ts_code"] == ts_code]
        if not hit.empty:
            return str(hit.iloc[0]["name"])
    return None


def publish_kline(ts_code: str, count: int = 250) -> dict:
    """读数据湖 -> 发布展示级 K线 JSON（形状对齐前端 /api/stock 的 kline 字段）。"""
    df = lake.load_daily_bar(ts_code)
    df = df[df["close"].notna()]  # 去掉停牌空行
    if df.empty:
        raise SystemExit(f"[publish] {ts_code} 数据湖里没有可用日线，请先 ingest-daily")
    df = df.iloc[-count:]
    kline = [
        {
            "t": str(r.trade_date),
            "o": round(float(r.open), 4),
            "h": round(float(r.high), 4),
            "l": round(float(r.low), 4),
            "c": round(float(r.close), 4),
            "v": int(round(float(r.volume or 0) / 100)),  # 股 -> 手（对齐 StockResult.kline.v）
        }
        for r in df.itertuples()
    ]
    payload = {
        "code": _digits(ts_code),
        "tsCode": ts_code,
        "name": _name_of(ts_code),
        "market": ts_code.split(".")[0],
        "adjust": "none",  # M0 阶段先发布未复权，复权口径留给引擎层
        "freq": "1d",
        "count": len(kline),
        "from": kline[0]["t"],
        "to": kline[-1]["t"],
        "publishedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kline": kline,
    }
    out_dir = config.PUBLISH_DIR / "kline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{payload['code']}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[publish-kline] {ts_code} {payload['from']}~{payload['to']} {len(kline)} 根 -> {out}")
    return payload


def cmd_publish_kline(args) -> None:
    publish_kline(bs.normalize_code(args.code), args.count)


def cmd_publish_all(args) -> None:
    for ts in lake.lake_codes():
        publish_kline(ts, args.count)


# ---------------- daily（launchd 入口） ----------------

def read_codes_file() -> list[str]:
    if not config.CODES_FILE.exists():
        raise SystemExit(f"缺少代码清单 {config.CODES_FILE}，一行一个代码，如 600519")
    return [re.sub(r"#.*", "", ln).strip() for ln in config.CODES_FILE.read_text(encoding="utf-8").splitlines() if re.sub(r"#.*", "", ln).strip()]


def cmd_daily(args) -> None:
    """codes.txt 全量流程：日历/列表 -> 逐票增量入库 -> 复权因子 -> 发布 K线。"""
    config.ensure_dirs()
    cal = bs.fetch_trade_dates("2010-01-01", (date.today() + timedelta(days=400)).isoformat())
    lake.write_meta("trade_calendar", cal)
    inst = bs.fetch_stock_list()
    if not inst.empty:
        lake.write_meta("instruments", inst)
    for raw in read_codes_file():
        ts = bs.normalize_code(raw)
        latest = lake.latest_trade_date(ts)
        start = ((date.fromisoformat(latest) + timedelta(days=1)).isoformat()
                 if latest else _default_start())
        end = bs.today_str()
        if start > end:
            print(f"[daily] {ts} 已是最新（{latest}），跳过入库")
        else:
            df = bs.fetch_daily(ts, start, end, adjust="none")
            if not df.empty:
                lake.upsert_daily_bar(df)
            adj = bs.fetch_adjust_factor(ts, start, end)
            if not adj.empty:
                lake.upsert_adjust_factor(adj)
            print(f"[daily] {ts} 增量入库 {start}~{end}")
        publish_kline(ts, 250)


def cmd_status(args) -> None:
    print(f"数据湖:   {config.DATA_DIR}")
    print(f"发布目录: {config.PUBLISH_DIR}")
    codes = lake.lake_codes()
    print(f"入库股票: {len(codes)} 只")
    for ts in codes:
        n = len(lake.load_daily_bar(ts))
        print(f"  {ts}  {n} 根  最新 {lake.latest_trade_date(ts)}")
    kdir = config.PUBLISH_DIR / "kline"
    files = sorted(kdir.glob("*.json")) if kdir.exists() else []
    print(f"已发布 K线 JSON: {len(files)} 个")
    for f in files:
        print(f"  {f.name}  {f.stat().st_size / 1024:.1f} KB")


def main() -> None:
    p = argparse.ArgumentParser(prog="qplat", description="个人量化平台阶段零 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, code=True, span=True):
        sp = sub.add_parser(name)
        if code:
            sp.add_argument("--code", required=True)
        if span:
            sp.add_argument("--start"); sp.add_argument("--end")
        sp.set_defaults(fn=fn)
        return sp

    add("ingest-daily", cmd_ingest_daily)
    add("ingest-adjust", cmd_ingest_adjust)
    add("ingest-calendar", cmd_ingest_calendar, code=False)
    add("ingest-list", cmd_ingest_list, code=False, span=False)

    sp = sub.add_parser("publish-kline"); sp.add_argument("--code", required=True); sp.add_argument("--count", type=int, default=250); sp.set_defaults(fn=cmd_publish_kline)
    sp = sub.add_parser("publish-all"); sp.add_argument("--count", type=int, default=250); sp.set_defaults(fn=cmd_publish_all)
    sp = sub.add_parser("daily"); sp.set_defaults(fn=cmd_daily)
    sp = sub.add_parser("status"); sp.set_defaults(fn=cmd_status)

    args = p.parse_args()
    config.ensure_dirs()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
