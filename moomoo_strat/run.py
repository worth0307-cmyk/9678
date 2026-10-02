"""moomoo 版 combo_overlay + impulse_wave：连接检查、回测、每日筛选。

    python moomoo_strat/run.py check                          # OpenD 连上没有、登录没有、额度、自选股分组
    python moomoo_strat/run.py backtest --group 美股           # 拿自选股分组回测 8 套规则
    python moomoo_strat/run.py screen   --group 美股           # 今天每只股票的指标状态和信号
    python moomoo_strat/run.py screen   --group 美股 --to-group 策略信号 --rule S2
                                                              # 并把符合 S2 的股票同步进 moomoo 分组「策略信号」
    python moomoo_strat/run.py backtest --csv-dir data --symbols BTCUSDT,ETHUSDT   # 不连 OpenD，读本地 CSV
    python moomoo_strat/run.py menu                           # 中文菜单（moomoo_tool.bat 双击进的就是这个）

详细说明见 moomoo_strat/README.md。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
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
        if a.group == "?":
            a.group = tag = pick_group(od)
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


LAST = HERE / "cache" / "last_group.txt"


def pick_group(od, prompt: str = "选哪个分组？输入编号后回车", custom_only: bool = False, default: str | None = None) -> str:
    """列出自选股分组、按编号选，不用打中文。直接回车 = 上次用的那个（或 default）"""
    g = od.groups()
    if custom_only:
        g = g[g["group_type"].astype(str).str.upper() == "CUSTOM"]
    names = list(g["group_name"])
    if not names:
        sys.exit("没有可选的分组" + ("（只列自定义分组，先在 App 里建一个）" if custom_only else ""))
    last = default
    if last is None:
        try:
            last = LAST.read_text(encoding="utf-8").strip()
        except OSError:
            last = None
    for n, name in enumerate(names, 1):
        print(f"  {n:2d}  {name}{'   ← 回车选这个' if name == last else ''}")
    while True:
        pick = input(f"{prompt}：").strip()
        if not pick and last in names:
            choice = last
            break
        if pick.isdigit() and 1 <= int(pick) <= len(names):
            choice = names[int(pick) - 1]
            break
    if default is None:
        try:
            LAST.parent.mkdir(exist_ok=True)
            LAST.write_text(choice, encoding="utf-8")
        except OSError:
            pass
    return choice


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
            if a.to_group == "?":
                a.to_group = pick_group(od, "同步进哪个分组？（只列自定义分组）", custom_only=True, default="策略信号")
            try:
                add, rem = od.set_group(a.to_group, codes, dry=True)
            except RuntimeError as e:
                sys.exit(str(e))
            print(f"\n「{a.to_group}」要同步成 {a.rule} 当前的 {len(codes)} 只：加入 {add or '无'}，移出 {rem or '无'}")
            if a.confirm and (add or rem):
                if input("确认修改这个分组？输入 y 回车确认，其他键取消：").strip().lower() != "y":
                    print("没有修改")
                    return
            od.set_group(a.to_group, codes)
        print("已同步")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="moomoo 版 combo_overlay + impulse_wave")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("menu")
    m.set_defaults(fn=cmd_menu)
    for name, fn in (("check", cmd_check), ("backtest", cmd_backtest), ("screen", cmd_screen)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--host", default="127.0.0.1")
        s.add_argument("--port", type=int, default=11111)
        s.add_argument("--group", help="moomoo 自选股分组名（App 里看到的名字，如「全部」「美股」或自己建的）；"
                                       "填 ? 就列出全部分组、按编号选")
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
            s.add_argument("--to-group", help="把符合 --rule 的股票同步进这个 moomoo **自定义**分组（要先在 App 里建好）；"
                                              "填 ? 就按编号选")
            s.add_argument("--rule", default="S3", choices=["S0", "S1", "S2", "S3", "S4"],
                           help="同步哪条规则（默认 S3：今天刚出五浪做多信号、或五浪多单还拿着的股票）")
            s.add_argument("--confirm", action="store_true", help="改分组之前先列出要加入 / 移出的股票，等你确认")
    return p


def run(argv: list[str]) -> None:
    a = parser().parse_args(argv)
    if getattr(a, "st", None):
        n, m = a.st.split(",")
        S.ST = (int(n), float(m))
    a.fn(a)


MENU = [
    ("1", "检查连接：OpenD、额度、自选股分组", ["check"]),
    ("2", "回测一个自选股分组（按编号选，回车 = 上次那个）", ["backtest", "--group", "?"]),
    ("3", "回测 79 只美股大盘股（样本外检验，清单 universe/us_large80.txt）", ["backtest", "--codes-file", "us_large80.txt"]),
    ("4", "今日筛选一个分组", ["screen", "--group", "?"]),
    ("5", "今日筛选，并把五浪做多信号（S3）同步进一个自定义分组（先预览，确认了才改）",
     ["screen", "--group", "?", "--to-group", "?", "--rule", "S3", "--confirm"]),
    ("8", "打开结果文件夹", None),
    ("9", "更新代码（下载最新版、装依赖）", None),
    ("0", "退出", None),
]


def cmd_menu(a) -> None:
    """中文菜单。moomoo_tool.bat 只负责找 python、下载代码，菜单的中文全在这里 ——
    cmd 读 UTF-8 的 bat 会把含中文的行读错位，放在 Python 里就没这个问题。
    退出码 99 = 让 bat 去更新代码（正在运行的程序不好自己覆盖自己）。"""
    while True:
        print("\n  moomoo 策略工具    combo_overlay + impulse_wave")
        print("  " + "-" * 52)
        print("  先打开 moomoo OpenD 并登录，窗口保持开着\n")
        for k, label, _ in MENU:
            print(f"  {k}  {label}")
        try:
            c = input("\n输入编号后回车：").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
        item = next((m for m in MENU if m[0] == c), None)
        if item is None:
            continue
        if c == "0":
            sys.exit(0)
        if c == "9":
            sys.exit(99)
        if c == "8":
            R.OUT.mkdir(exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(R.OUT)
            else:
                print(R.OUT)
            continue
        try:
            run(item[2])
        except SystemExit as e:
            if e.code not in (None, 0):
                print(e.code if isinstance(e.code, str) else f"（退出码 {e.code}）")
        except KeyboardInterrupt:
            print("\n已中断")
        except Exception:
            traceback.print_exc()
            print("\n出错了：把上面这段截图发给我")
        print(f"\n报告和表格在 {R.OUT}")
        input("按回车回到菜单…")


def main() -> None:
    try:                                   # Windows 上重定向到文件时默认 GBK，⬆ 之类的字符会报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
