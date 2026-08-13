# 数据导出 Runbook（6 币种）

沿用你现在的流程，只是把单币扩成 6 个币。

> Claude Code 这个会话的出口策略封了 `api.binance.com` / `fapi.binance.com`
> （还有 coingecko / kraken / coinbase / bybit / okx，全部 403），所以导出必须在 VPS 上跑。

---

## 1. VPS 上导出

```bash
ssh vpn-sg
cd ~/VI-Dashboard && git fetch origin main && git checkout main && git reset --hard origin/main
```

然后直接贴这一段（不依赖本仓库的任何文件）：

```bash
for S in BTCUSDT BNBUSDT ETHUSDT HYPEUSDT SOLUSDT TAOUSDT; do
  echo "=== $S"
  python3 backend/tools/export_klines.py --symbol "$S" \
    --market futures --intervals 1h,4h,1d --start 2024-01-01 --out ./exports \
    || echo "  !! $S 失败，继续下一个"
done
ls -1 ./exports
```

跑完应该是 **18 个文件**（6 币 × 3 周期），命名形如 `ETHUSDT_4h_2024-01-01_to_now.csv`。

`--market futures`、`--intervals 1h,4h,1d`、`--start 2024-01-01` 就是 `export_klines.py` 的默认值，
也就是你之前裸跑 `python3 backend/tools/export_klines.py` 得到 BTCUSDT 那批数据的口径（**U 本位合约**）。
这里写全只是为了明确，不是改了什么。

> 如果本仓库已经在 VPS 上，也可以直接 `bash /path/to/9678/scripts/fetch_universe.sh`，效果一样。

---

## 2. 拉回本地

```powershell
scp -r vpn-sg:~/VI-Dashboard/exports "$env:USERPROFILE\Desktop\VI数据"
```

（原来那条是 `...\Desktop\BTCUSDT数据`；现在里面是 6 个币了，换个名字免得误会。）

---

## 3. 导入本仓库

```bash
python -m vibt.ingest "$env:USERPROFILE\Desktop\VI数据"
```

`ingest` 会做四件事：

1. 归一化命名 → `data/{SYMBOL}_{1h,4h,1d}.csv`
2. 逐币逐周期报告：K 线根数、真实起止时间、**缺口数、不可能的 OHLC、非正价格**
3. 用 1h 重采样交叉校验 4h / 1d 是否自洽
4. 打印最终可加载的币种列表

先看一遍报告再往下跑。想先看不写盘就加 `--dry-run`。

---

## 4. 跑池化研究

```bash
python -m pytest tests/ -q          # 16 个测试，含无前视检验
python scripts/12_multi_symbol.py   # 六币种池化的 VI+ 双突破事件研究
```

这一步才是加币种的意义：现在单 BTC 只有 61 个事件，什么都证明不了；6 个币有 300+ 个。
脚本会在三种通道口径（固定 / 滚动分位 / 滚动区间）下**池化统计**，并给出**逐币一致性检查**。

**怎么读结果：** 如果"上破后继续上涨"在 6 个币上都成立，它就不是 BTC 的偶然；
如果只在 BTC 上成立、其他 5 个币各说各话，那它本来就是噪声。

---

## 预期会遇到的两件事

**HYPE / TAO 的历史短于 2024-01-01。** HYPE 是 2024 年 11 月底才上线的。
`ingest` 会如实报告每个币的真实起始时间，池化统计自动只用各自有数据的区间。
但要注意：**HYPE 的样本基本只覆盖了 2025-2026 的下跌，不覆盖 2024 那波牛市**，
而第 5 节已经证明这个信号的盈亏符号跟着大趋势走——所以别把 HYPE 的结果和 BTC 的直接平均。

**Ticker 可能对不上。** HYPEUSDT / TAOUSDT 在币安 U 本位合约上的确切交易对名称，
我在这个会话里连不上币安没法验证。哪个失败了就看 Dashboard 里 `symbols` 配置用的字符串，
或者直接改上面循环里的名字。
