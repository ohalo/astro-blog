#!/usr/bin/env python3
"""Stock valuation calculator for PE, PB, PEG and FCFF-DCF scenarios.

All monetary inputs are in CNY (or the reporting currency of the company).
Share counts are in shares. Prices are returned per share.

Examples:
  python valuation_calc.py pe --shares 431404377 --profit 784000000 --pe-low 20 --pe-high 25 --current 34.74
  python valuation_calc.py pb --navps 4.205 --pb-low 6 --pb-high 8 --current 34.74
  python valuation_calc.py peg --pe 30.83 --growth 31.6
  python valuation_calc.py dcf --shares 431404377 --fcff 600000000 700000000 800000000 \
      --wacc 0.10 --g 0.025 --net-debt 340000000 --minority 0 --current 34.74
"""

import argparse
import sys
from typing import List


def _fmt_money(value: float) -> str:
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.2f}万"
    return f"{value:.2f}"


def _fmt_price(value: float) -> str:
    return f"{value:.2f}"


def _print_upside(current: float | None, low: float, high: float) -> None:
    if current is None:
        return
    low_pct = (low / current - 1) * 100
    high_pct = (high / current - 1) * 100
    print(f"相对现价 {_fmt_price(current)}：低端空间 {low_pct:+.1f}%，高端空间 {high_pct:+.1f}%")


def cmd_pe(args: argparse.Namespace) -> int:
    if args.shares <= 0:
        raise ValueError("shares must be positive")
    if args.profit == 0:
        raise ValueError("profit must be non-zero; PE method does not apply to zero profit")
    if args.pe_low <= 0 or args.pe_high <= 0:
        raise ValueError("PE must be positive")
    if args.pe_low > args.pe_high:
        args.pe_low, args.pe_high = args.pe_high, args.pe_low

    eps = args.profit / args.shares
    low = eps * args.pe_low
    high = eps * args.pe_high
    mid = eps * (args.pe_low + args.pe_high) / 2

    print(f"EPS = {_fmt_money(args.profit)} / {args.shares:,} 股 = {eps:.4f} 元/股")
    print(f"合理股价区间 = {low:.2f} ~ {high:.2f} 元")
    print(f"区间中值 = {mid:.2f} 元")
    _print_upside(args.current, low, high)
    return 0


def cmd_pb(args: argparse.Namespace) -> int:
    if args.pb_low <= 0 or args.pb_high <= 0:
        raise ValueError("PB must be positive")
    if args.pb_low > args.pb_high:
        args.pb_low, args.pb_high = args.pb_high, args.pb_low

    low = args.navps * args.pb_low
    high = args.navps * args.pb_high
    mid = args.navps * (args.pb_low + args.pb_high) / 2

    print(f"BVPS = {args.navps:.4f} 元/股")
    print(f"合理股价区间 = {low:.2f} ~ {high:.2f} 元")
    print(f"区间中值 = {mid:.2f} 元")
    _print_upside(args.current, low, high)
    return 0


def cmd_peg(args: argparse.Namespace) -> int:
    if args.growth <= 0:
        raise ValueError("growth must be positive for PEG")
    if args.pe <= 0:
        raise ValueError("PE must be positive")

    peg = args.pe / args.growth
    print(f"PEG = PE {args.pe:.2f} / 增速 {args.growth:.2f}% = {peg:.2f}")
    if peg < 0.8:
        print("PEG 偏低：成长与估值的匹配度相对占优，但仍需验证增速可持续性")
    elif peg <= 1.2:
        print("PEG 接近1：成长股常见合理区间，不算明显便宜")
    else:
        print("PEG 偏高：估值对增长预期较敏感，需警惕增速下修")
    return 0


def cmd_dcf(args: argparse.Namespace) -> int:
    fcff: List[float] = args.fcff
    if not fcff:
        raise ValueError("fcff list cannot be empty")
    if args.wacc <= 0:
        raise ValueError("WACC must be positive")
    if args.g >= args.wacc:
        raise ValueError("g must be smaller than WACC")
    if args.shares <= 0:
        raise ValueError("shares must be positive")

    pv_explicit = sum(cf / ((1 + args.wacc) ** (i + 1)) for i, cf in enumerate(fcff))
    terminal_value = fcff[-1] * (1 + args.g) / (args.wacc - args.g)
    pv_terminal = terminal_value / ((1 + args.wacc) ** len(fcff))
    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - args.net_debt - args.minority
    per_share = equity_value / args.shares

    print(f"显性期现值 = {_fmt_money(pv_explicit)}")
    print(f"终值现值 = {_fmt_money(pv_terminal)}")
    print(f"终值占企业价值比重 = {pv_terminal / enterprise_value * 100:.1f}%")
    print(f"企业价值 EV = {_fmt_money(enterprise_value)}")
    print(f"股权价值 = {_fmt_money(equity_value)}")
    print(f"每股价值 = {per_share:.2f} 元")

    if pv_terminal / enterprise_value > 0.70:
        print("警示：终值占比超过70%，估值对长期增长和WACC高度敏感，应给出多情景而非单一结论")
    if args.current is not None:
        pct = (per_share / args.current - 1) * 100
        print(f"相对现价 {_fmt_price(args.current)}：空间 {pct:+.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate stock intrinsic value scenarios")
    sub = parser.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("pe", help="Price / expected net profit scenario")
    pe.add_argument("--shares", type=float, required=True, help="总股本，单位股")
    pe.add_argument("--profit", type=float, required=True, help="归母净利润，单位元")
    pe.add_argument("--pe-low", type=float, required=True, help="合理PE下限")
    pe.add_argument("--pe-high", type=float, required=True, help="合理PE上限")
    pe.add_argument("--current", type=float, help="当前股价，可选")
    pe.set_defaults(func=cmd_pe)

    pb = sub.add_parser("pb", help="Price / book value scenario")
    pb.add_argument("--navps", type=float, required=True, help="每股净资产 BVPS，单位元/股")
    pb.add_argument("--pb-low", type=float, required=True, help="合理PB下限")
    pb.add_argument("--pb-high", type=float, required=True, help="合理PB上限")
    pb.add_argument("--current", type=float, help="当前股价，可选")
    pb.set_defaults(func=cmd_pb)

    peg = sub.add_parser("peg", help="PEG sanity check")
    peg.add_argument("--pe", type=float, required=True, help="当前或合理PE")
    peg.add_argument("--growth", type=float, required=True, help="预期净利润增速，单位%")
    peg.set_defaults(func=cmd_peg)

    dcf = sub.add_parser("dcf", help="FCFF DCF scenario")
    dcf.add_argument("--shares", type=float, required=True, help="总股本，单位股")
    dcf.add_argument("--fcff", type=float, nargs="+", required=True, help="显性期FCFF序列，单位元")
    dcf.add_argument("--wacc", type=float, required=True, help="折现率，例如 0.10")
    dcf.add_argument("--g", type=float, required=True, help="永续增长率，例如 0.025")
    dcf.add_argument("--net-debt", type=float, default=0.0, help="净债务=有息负债-现金，默认0")
    dcf.add_argument("--minority", type=float, default=0.0, help="少数股东权益，默认0")
    dcf.add_argument("--current", type=float, help="当前股价，可选")
    dcf.set_defaults(func=cmd_dcf)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
