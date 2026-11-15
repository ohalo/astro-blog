# platform/ —— 个人量化平台（阶段零：无后端）

对应规划书 `docs/quant-platform-plan.md`。本目录是平台的计算内核：**纯 Python，零 Web 依赖**，
阶段零以 CLI 形式把聚合结果发布为 `public/data/*.json` 静态文件，随博客一起部署到 Vercel。

```
baostock ──采集──▶ quant-data/*.parquet ──DuckDB 聚合──▶ public/data/kline/*.json ──git push──▶ Vercel
   (数据源适配层)      (数据湖, gitignore)      (发布器 qplat publish)      (前端 KLine.astro 直读)
```

## 环境重建（新机器照抄一遍即可）

```bash
cd platform
export UV_CACHE_DIR="$PWD/.uv-cache"          # 缓存放进工作区，沙箱/无 home 写权限也能跑
uv venv .venv --python /Users/halo123456/.local/bin/python3.12
uv pip install --python .venv/bin/python baostock pandas pyarrow duckdb numpy
```

## 命令一览（`./venv` 内执行，`.venv/bin/python -m cli.qplat <cmd>`）

| 命令 | 作用 |
|---|---|
| `ingest-daily --code 600519 [--start --end]` | 日线（未复权）拉取入库，默认近 10 年，幂等 upsert |
| `ingest-adjust --code 600519` | 复权因子（除息日粒度）入库 |
| `ingest-calendar` / `ingest-list` | 交易日历 / 证券主表入库 |
| `publish-kline --code 600519 [--count 250]` | 发布单票 K线 JSON 到 `public/data/kline/` |
| `publish-all [--count 250]` | 发布数据湖内全部股票 |
| `daily` | **launchd 入口**：日历/主表 + codes.txt 逐票增量入库 + 发布 |
| `status` | 数据湖与发布产物体检 |

自选管理：编辑 `codes.txt`（一行一个代码）。

## 每日自动更新（launchd）

```bash
# 1) 先手动验证一次（会 git commit + push，确认无误再挂定时）
zsh platform/runbooks/daily.sh

# 2) 挂到 launchd（工作日 17:10）
cp platform/runbooks/com.qplat.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.qplat.daily.plist

# 卸载：launchctl unload ~/Library/LaunchAgents/com.qplat.daily.plist
# 日志：platform/runs/launchd.{out,err}.log
```

不想每天自动 push，把 `daily.sh` 里 `git push` 行注释掉，改为攒着手动推。

## 数据湖布局（gitignore，不进库）

```
quant-data/
├── daily_bar/year=YYYY/data.parquet     # (ts_code, trade_date) 去重, 新数据为准
├── adjust_factor/year=YYYY/...          # 除息日复权因子
└── meta/{trade_calendar,instruments}.parquet
```

环境变量：`QUANT_DATA_DIR` / `QUANT_PUBLISH_DIR` / `QUANT_CODES_FILE` 可覆盖三个路径（为上云预留）。

## 发布 JSON 契约

`public/data/kline/{code}.json`：`kline[].{t,o,h,l,c,v}` 与 `src/pages/api/stock.ts` 的
`StockResult.kline` 对齐（v 单位=手），前端 `src/components/dash/KLine.astro` 直读渲染。
阶段零发布**未复权**价（`adjust:"none"`）；复权口径计算留给引擎层，勿在发布器里算。

## 实测踩坑备忘（2026-11）

- baostock `query_adjust_factor` 实际字段是 `code/dividOperateDate/foreAdjustFactor/backAdjustFactor/adjustFactor`，与文档示例不一致 —— 适配层已消化，上层别依赖原始字段名。
- baostock 每次查询需处于 login 会话内；适配层每个 `fetch_*` 自带 login/logout，批量任务别在外层再套 `session()` 之外裸调。
- duckdb ≥1.4（nanobind 版）`execute().df` 属性不再返回 DataFrame，用 `fetch_df()/fetchdf()` —— `lake.query()` 已兼容双版本。
- venv 里 pandas 解析到了 3.x / python-dateutil 4.x：baostock 目前兼容，若上游炸了先 `uv pip install 'pandas<3'` 降级。
- `bun add` 撞 home 缓存权限时用 `--cache-dir ./node_modules/.bun-cache`。
- baostock 当日日线约 17:00 后就绪；17:10 跑 `daily` 拿得到当天数据，周末跑等于空转（脚本会正确跳过）。

## 里程碑状态

- [x] M0 数据管道 + K线看板（本目录 + `/dash` 页 + `KLine.astro`）
- [ ] M0 余量：全市场日线入库（codes.txt 扩到全量 or 加 `ingest-market` 命令）、复权展示口径
- [ ] M1 因子看板（`quant/factor/analysis.py`：IC/分层/衰减，从 `scripts/article_core_*.py` 抽公共算子）
- [ ] M2 回测引擎（`quant/engine/`：T+1/涨跌停/停牌撮合）
- [ ] M3 模拟实盘（`quant/sim/`）
- [ ] M4 用户系统 + FastAPI + 上云（触发式，见规划书 0.5/12 节）
