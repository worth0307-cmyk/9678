"""数据：从 moomoo OpenD 拉自选股分组和前复权日 K，或者从本地 CSV 读（离线测试用）。

OpenD 是 moomoo 官方的本地网关程序，Python 只跟它说话（默认 127.0.0.1:11111），
登录、行情权限都在 OpenD 里。安装和登录见 README.md。

历史 K 线有额度：每 30 天内能拉的**不同股票**只数有上限（具体看账户等级，check 会打印），
同一只股票 30 天内重复拉不再扣额度。频率上限是每 30 秒 60 次请求，这里每次请求间隔 0.6 秒。
"""

from __future__ import annotations

import datetime as dt
import logging
import socket
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"


# ---------------------------------------------------------------- OpenD

class OpenD:
    """OpenQuoteContext 的薄封装：用 with 自动关闭连接"""

    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        try:
            import moomoo as mm
        except ImportError as e:
            raise SystemExit("没装 moomoo-api：pip install -r moomoo_strat/requirements.txt") from e
        # 先试一下端口：OpenD 没开时，SDK 的 OpenQuoteContext 会每 6 秒重连一次、永远不返回，看上去就是卡住了
        try:
            socket.create_connection((host, port), timeout=3).close()
        except OSError as e:
            raise SystemExit(
                f"连不上 OpenD（{host}:{port}）：{e}\n"
                "  · OpenD 打开并登录了吗？注意 moomoo 桌面版 / App 不是 OpenD，OpenD 是另一个程序\n"
                "  · OpenD 要和这个 Python 程序在同一台电脑上（OpenD 在 Windows、程序在 VPS 上是连不到的）\n"
                "  · 改过端口的话加 --port") from e
        try:                                    # SDK 每次连接、断开都往屏幕打一行 INFO，只留警告和错误
            from moomoo.common.ft_logger import logger
            logger.console_level = logging.WARNING
        except Exception:
            pass
        self.mm = mm
        self.ctx = mm.OpenQuoteContext(host=host, port=port)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.ctx.close()

    def _ok(self, ret, data, what: str):
        if ret != self.mm.RET_OK:
            raise RuntimeError(f"{what} 失败：{data}")
        return data

    def state(self) -> dict:
        return self._ok(*self.ctx.get_global_state(), "get_global_state")

    def quota(self) -> tuple[int, int, set]:
        """(已用, 剩余, 30 天内已经拉过的代码) —— 拉过的再拉不扣额度"""
        used, remain, detail = self._ok(*self.ctx.get_history_kl_quota(get_detail=True), "get_history_kl_quota")
        return used, remain, {d.get("code") for d in (detail or [])}

    def groups(self) -> pd.DataFrame:
        return self._ok(*self.ctx.get_user_security_group(self.mm.UserSecurityGroupType.ALL),
                        "get_user_security_group")

    def watchlist(self, group: str) -> pd.DataFrame:
        """自选股分组里的股票：code（如 US.AAPL、HK.00700）、name 等"""
        df = self._ok(*self.ctx.get_user_security(group), f"读取自选股分组「{group}」")
        # 分组里可能有指数、期权、期货之类，只留股票和 ETF
        if "stock_type" in df.columns:
            df = df[df["stock_type"].astype(str).isin(["STOCK", "ETF"])]
        return df.reset_index(drop=True)

    def set_group(self, group: str, codes: list[str]) -> tuple[list[str], list[str]]:
        """把一个**自定义**分组同步成 codes：多的移出、少的加入。返回 (加入, 移出)。"""
        Op = self.mm.ModifyUserSecurityOp
        cur = set(self.watchlist(group)["code"]) if group in set(self.groups()["group_name"]) else set()
        add, rem = sorted(set(codes) - cur), sorted(cur - set(codes))
        if rem:
            self._ok(*self.ctx.modify_user_security(group, Op.MOVE_OUT, rem), f"从「{group}」移出")
        if add:
            self._ok(*self.ctx.modify_user_security(group, Op.ADD, add), f"加入「{group}」")
        return add, rem

    def daily(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        """前复权日 K，自动翻页。列：open high low close volume，索引是交易日"""
        mm = self.mm
        end = end or dt.date.today().isoformat()
        parts, key = [], None
        while True:
            ret, data, key = self.ctx.request_history_kline(
                code, start=start, end=end, ktype=mm.KLType.K_DAY, autype=mm.AuType.QFQ,
                fields=[mm.KL_FIELD.ALL], max_count=1000, page_req_key=key)
            time.sleep(0.6)
            self._ok(ret, data, f"拉 {code} 日 K")
            parts.append(data)
            if key is None:
                break
        df = pd.concat(parts, ignore_index=True)
        df.index = pd.to_datetime(df["time_key"]).dt.normalize()
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df[~df.index.duplicated(keep="last")].sort_index()


def load_moomoo(od: OpenD, codes: list[str], start: str, refresh: bool = False, log=print) -> dict[str, pd.DataFrame]:
    """拉一组股票的日 K，按天缓存在 cache/ 里。

    前复权价格在每次除权后整段都会变，所以缓存不做增量拼接：今天拉过就用缓存，不是今天的就整段重拉
    （同一只股票 30 天内重拉不扣额度）。
    """
    CACHE.mkdir(exist_ok=True)
    today = dt.date.today()
    out = {}
    for n, code in enumerate(codes, 1):
        f = CACHE / f"{code}_D.csv"
        fresh = f.exists() and dt.date.fromtimestamp(f.stat().st_mtime) == today
        if fresh and not refresh:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(df) and df.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=10):
                out[code] = df[df.index >= pd.Timestamp(start)]
                continue
        try:
            df = od.daily(code, start)
        except RuntimeError as e:
            log(f"  [{n}/{len(codes)}] {code} 跳过：{e}")
            continue
        df = df[df["close"] > 0]
        df.to_csv(f)
        out[code] = df
        log(f"  [{n}/{len(codes)}] {code} {len(df)} 根  {df.index[0].date()} ~ {df.index[-1].date()}")
    return out


# ---------------------------------------------------------------- 本地 CSV

_COLS = {
    "time_key": "date", "date": "date", "datetime": "date", "日期": "date", "日期时间(北京)": "date",
    "open": "open", "开盘": "open", "high": "high", "最高": "high",
    "low": "low", "最低": "low", "close": "close", "收盘": "close", "volume": "volume", "成交量": "volume",
}


def load_csv(path: Path) -> pd.DataFrame:
    """一个文件一只股票，列名认 moomoo 导出 / 本仓库 data/ / 常见英文三种写法"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={c: _COLS[c] for c in df.columns if c in _COLS})
    df.index = pd.to_datetime(df.pop("date")).dt.normalize()
    if "volume" not in df:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_dir(d: Path, names: list[str] | None, start: str) -> dict[str, pd.DataFrame]:
    """目录里的 CSV：names 给了就只读这些（文件名以它开头，*_1d.csv.gz 优先）"""
    files = sorted(list(d.glob("*.csv")) + list(d.glob("*.csv.gz")))
    out = {}
    for nm in names or sorted({f.name.split("_")[0].split(".csv")[0] for f in files}):
        cand = [f for f in files if f.name.startswith(nm)]
        cand.sort(key=lambda f: (("_1d" not in f.name), f.name))
        if not cand:
            print(f"  {nm}：{d} 里没有对应的 CSV，跳过")
            continue
        df = load_csv(cand[0])
        out[nm] = df[df.index >= pd.Timestamp(start)]
    return out
