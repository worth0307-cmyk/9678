"""moomoo 版 combo_overlay + impulse_wave：连接检查、回测、每日筛选。

    python moomoo_strat/run.py check                          # OpenD 连上没有、登录没有、额度、自选股分组
    python moomoo_strat/run.py backtest --group 美股           # 拿自选股分组回测 8 套规则
    python moomoo_strat/run.py screen   --group 美股           # 今天每只股票的指标状态和信号
    python moomoo_strat/run.py screen   --group 美股 --to-group 策略信号 --rule S2
                                                              # 并把符合 S2 的股票同步进 moomoo 分组「策略信号」
    python moomoo_strat/run.py backtest --csv-dir data --symbols BTCUSDT,ETHUSDT   # 不连 OpenD，读本地 CSV

详细说明见 moomoo_strat/README.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    # 直接 python moomoo_strat/run.py：把仓库根目录放进路径，按包导入；
    # 同时去掉本目录，免得 data.py 之类的名字盖住别的库
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    sys.path.insert(0, str(HERE.parent))
    __package__ = "moomoo_strat"

import pandas as pd  # noqa: E402

from moomoo_strat import data as DA, report as R, strategy as S  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.unicode.east_asian_width", True)


def load(a) -> tuple[dict, dict, str]:
    """返回 (代码 → 日 K, 代码 → 名称, 标签)"""
    if a.csv_dir:
        names = a.symbols.split(",") if a.symbols else None
        dfs = DA.load_dir(Path(a.csv_dir), names, a.start)
        return dfs, {}, Path(a.csv_dir).name
    tag = a.group or "codes"
    if getattr(a, "codes_file", None):
        f = Path(a.codes_file)
        if not f.exists():          # 不在仓库根目录跑时，也认相对仓库根目录或 universe/ 的写法
            f = next((c for c in (HERE.parent / a.codes_file, HERE / "universe" / a.codes_file) if c.exists()), f)
        if not f.exists():
            sys.exit(f"找不到代码清单文件 {a.codes_file}")
        a.codes = ",".join(ln.split("#")[0].strip() for ln in f.read_text(encoding="utf-8").splitlines()
                           if ln.split("#")[0].strip())
        tag = f.stem
    if not (a.group or a.codes):
        sys.exit("要么给 --group 自选股分组名，要么 --codes US.AAPL,US.MSFT 或 --codes-file，要么 --csv-dir 离线目录")
    with DA.OpenD(a.host, a.port) as od:
        if a.group:
            wl = od.watchlist(a.group)
            if wl.empty:
                sys.exit(f"自选股分组「{a.group}」里没有股票 / ETF（指数、期权等会被略过）")
            codes, names = list(wl["code"]), dict(zip(wl["code"], wl["name"]))
        else:
            codes, names = a.codes.split(","), {}
        used, remain, seen = od.quota()
        new = [c for c in codes if c not in seen]
        print(f"历史 K 线额度：已用 {used}，剩余 {remain}；这次 {len(codes)} 只里 {len(new)} 只是 30 天内没拉过的，"
              f"会扣额度")
        if len(new) > remain:
            sys.exit(f"额度不够：要 {len(new)}，只剩 {remain}。等额度恢复（按 30 天滚动），或者少拉几只")
        print(f"拉 {len(codes)} 只的前复权日 K（{a.start} 起）…")
        dfs = DA.load_moomoo(od, codes, a.start, refresh=a.refresh)
    return dfs, names, tag


def cmd_check(a) -> None:
    with DA.OpenD(a.host, a.port) as od:
        st = od.state()
        print(f"OpenD 版本 {st.get('server_ver')}   行情登录 {'✓' if st.get('qot_logined') in ('1', 1, True) else '✗ 没登录'}"
              f"   美股 {st.get('market_us')}   港股 {st.get('market_hk')}")
        used, remain, _ = od.quota()
        print(f"历史 K 线额度：已用 {used}，剩余 {remain}")
        g = od.groups()
        print(f"\n自选股分组（{len(g)} 个）：")
        for _, r in g.iterrows():
            print(f"  {r['group_name']}  [{r['group_type']}]")
        if a.group:
            wl = od.watchlist(a.group)
            print(f"\n「{a.group}」里的股票 / ETF：{len(wl)} 只")
            print(wl[["code", "name"]].to_string(index=False))


def cmd_backtest(a) -> None:
    dfs, names, tag = load(a)
    if not dfs:
        sys.exit("没有数据")
    ann = S.ann_of(dfs)
    print(f"算指标（{len(dfs)} 只，一年按 {ann} 根）…")
    feats = R.prepare(dfs, names, ann)
    print("回测和零假设（几十只股票大约一两分钟）…\n")
    print(R.backtest(feats, names, ann, a.cost_bps * 1e-4, tag, null_reps=a.null_reps))


def cmd_screen(a) -> None:
    dfs, names, tag = load(a)
    ann = S.ann_of(dfs)
    feats = R.prepare(dfs, names, ann)
    t, path = R.screen(feats, names, a.cost_bps * 1e-4, tag)
    show = t.drop(columns=["五浪"])
    print(show.to_string(index=False))
    fades = t[t["五浪"] != ""]
    if len(fades):
        print("\n五浪信号：")
        for _, r in fades.iterrows():
            print(f"  {r['代码']} {r['名称']}：{r['五浪']}")
    print(f"\n已存 {path}")
    print("D / W = 推动浪趋势（中档），规则用的 W 是上一个走完的周；「W本周至今」是 TradingView 小标签实时显示的那个，"
          "周五收盘才定。\nST线 = SuperTrend 那条线，多头时就是止损位。失效价 = 价格越过这里，这套数浪就作废。"
          "一年区间位置 = 现价在一年最高最低之间的百分位（斐波那契 0~1）。")
    if a.to_group:
        if a.csv_dir:
            sys.exit("--to-group 要连 OpenD")
        codes = list(t.loc[t[a.rule] == "✓", "代码"])
        with DA.OpenD(a.host, a.port) as od:
            if a.to_group not in set(od.groups()["group_name"]):
                sys.exit(f"moomoo 里没有叫「{a.to_group}」的自选股分组 —— 先在 App 里建一个空分组再跑")
            add, rem = od.set_group(a.to_group, codes)
        print(f"\n「{a.to_group}」已同步为 {a.rule} 当前持有的 {len(codes)} 只：加入 {add or '无'}，移出 {rem or '无'}")


def main() -> None:
    try:                                   # Windows 上重定向到文件时默认 GBK，⬆ 之类的字符会报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description="moomoo 版 combo_overlay + impulse_wave")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("backtest", cmd_backtest), ("screen", cmd_screen)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--host", default="127.0.0.1")
        s.add_argument("--port", type=int, default=11111)
        s.add_argument("--group", help="moomoo 自选股分组名（App 里看到的名字，如「全部」「美股」或自己建的）")
        if name != "check":
            s.add_argument("--codes", help="不用分组时直接给代码，逗号分隔：US.AAPL,HK.00700")
            s.add_argument("--codes-file", help="代码清单文件，一行一个，# 后面是注释（例：moomoo_strat/universe/us_large80.txt）")
            s.add_argument("--start", default="2015-01-01", help="从哪天开始拉（默认 2015-01-01）")
            s.add_argument("--refresh", action="store_true", help="忽略今天的缓存，重新拉")
            s.add_argument("--cost-bps", type=float, default=5.0, help="每边费用 bps（美股 5，港股建议 15）")
            s.add_argument("--st", default="10,3", help="SuperTrend 的 ATR 周期,倍数（默认 10,3，和 TradingView 图上设置一致才对得上）")
            s.add_argument("--csv-dir", help="离线：从这个目录读 CSV，不连 OpenD")
            s.add_argument("--symbols", help="离线时只读这些（文件名开头），逗号分隔")
        if name == "backtest":
            s.add_argument("--null-reps", type=int, default=300, help="随机择时零假设的次数")
        if name == "screen":
            s.add_argument("--to-group", help="把符合 --rule 的股票同步进这个 moomoo 自定义分组（要先在 App 里建好）")
            s.add_argument("--rule", default="S2", choices=["S0", "S1", "S2", "S4"])
    a = p.parse_args()
    if getattr(a, "st", None):
        n, m = a.st.split(",")
        S.ST = (int(n), float(m))
    a.fn(a)


if __name__ == "__main__":
    main()
