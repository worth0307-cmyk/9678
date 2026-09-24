"""推动浪指标（pine/impulse_wave_12345.pine）场景推演的 Python 逐行移植与验证.

    python scripts/59_impulse_scenario_check.py

Pine 没有编译器可用，所以把 ZigZag、旧版推演、新版场景引擎逐行搬到 Python，
在仓库里的真实数据上跑：

  A  覆盖率：把历史上随机 60 根逐一当作「最后一根」，看有多大比例画得出推演。
     旧版只认得「正在走浪5」一个位置（外加刚确认 1-5 之后一小段），新版对
     周期 1-2-3-4-5-A-B-C 八个位置逐一检验。
  B  路径自检：时间递增、每段方向正确、浪2 不破起点、浪4 不进浪1、
     B 不越过浪5、C 不破推动浪起点、未来不超过 480 根。
  C  最新一根的推演路径，每段的价格变化 / 根数 / 斜率。

周线由日线重采样（W-MON），和 TradingView 的周线对齐方式可能差一两天。
"""

from __future__ import annotations

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from vibt import data as D

def atr_w(h,l,c,n=14):
    pc=np.r_[np.nan,c[:-1]]; tr=np.nanmax(np.c_[h-l,np.abs(h-pc),np.abs(l-pc)],axis=1)
    o=np.full(len(c),np.nan)
    if len(c)>n:
        o[n]=np.nanmean(tr[1:n+1])
        for i in range(n+1,len(c)): o[i]=(o[i-1]*(n-1)+tr[i])/n
    return o

def load(sym, tf):
    if tf == '1w':
        d = D.load('1d', symbol=sym)
        return d.resample('W-MON', label='left', closed='left').agg(
            {'open':'first','high':'max','low':'min','close':'last'}).dropna()
    return D.load(tf, symbol=sym)

AUTO = {'1w': (0.5,1.0,1.5), '1d': (2,4,8), '4h': (2,4,8), '1h': (3,6,12)}

def zz(df, mult, upto=None):
    """ZigZag，跑到第 upto 根（含）为止 —— 用来模拟「在历史上某一根是最后一根」"""
    H,L_,C = df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy()
    A = atr_w(H,L_,C); N = len(C) if upto is None else upto+1
    pv=[]; d=0; ext=np.nan; ebi=-1; e2=np.nan; e2bi=-1
    for i in range(N):
        if np.isnan(A[i]): continue
        thr=A[i]*mult
        if d==0:
            if np.isnan(ext): ext,ebi,e2,e2bi=H[i],i,L_[i],i
            else:
                if H[i]>ext: ext,ebi=H[i],i
                if L_[i]<e2: e2,e2bi=L_[i],i
                if C[i]<ext-thr: pv.append((ebi,ext,True)); d,ext,ebi=-1,L_[i],i
                elif C[i]>e2+thr: pv.append((e2bi,e2,False)); d,ext,ebi=1,H[i],i
        elif d==1:
            if H[i]>ext: ext,ebi=H[i],i
            if C[i]<ext-thr: pv.append((ebi,ext,True)); d,ext,ebi=-1,L_[i],i
        else:
            if L_[i]<ext: ext,ebi=L_[i],i
            if C[i]>ext+thr: pv.append((ebi,ext,False)); d,ext,ebi=1,H[i],i
    return dict(pv=pv, d=d, ext=ext, ebi=ebi, N=N, close=C[N-1])

def old_draws(z, need5=True, ovTol=0.0, maxSpan=300):
    pv,d,N = z['pv'],z['d'],z['N']
    # (b) 正式：最近一组合规 1-5 的终点仍在未来
    for k in range(len(pv), 5, -1):
        p=pv[k-6:k]; isH=[x[2] for x in p]; bi=[x[0] for x in p]; pr=[x[1] for x in p]
        bull=isH==[False,True,False,True,False,True]; bear=isH==[True,False,True,False,True,False]
        if not(bull or bear): continue
        if any(bi[j+1]<=bi[j] for j in range(5)) or (bi[5]-bi[0])>maxSpan: continue
        l1,l3,l5=abs(pr[1]-pr[0]),abs(pr[3]-pr[2]),abs(pr[5]-pr[4])
        if not (pr[2]>pr[0] if bull else pr[2]<pr[0]) or (l3<l1 and l3<l5): continue
        if not (pr[4]>pr[1]-l1*ovTol if bull else pr[4]<pr[1]+l1*ovTol): continue
        if need5 and not (pr[5]>pr[3] if bull else pr[5]<pr[3]): continue
        if bi[5] + max(1,round((bi[5]-bi[0])*0.5)) >= N-1: return "正式"
        break
    # (a) 待定
    if len(pv) >= 5:
        p=pv[-5:]; isH=[x[2] for x in p]; pr=[x[1] for x in p]; bi=[x[0] for x in p]
        bull=isH==[False,True,False,True,False]; bear=isH==[True,False,True,False,True]
        if bull or bear:
            l1=abs(pr[1]-pr[0])
            ok = all(bi[j+1]>bi[j] for j in range(4)) and (pr[2]>pr[0] if bull else pr[2]<pr[0]) \
                 and (pr[4]>pr[1]-l1*ovTol if bull else pr[4]<pr[1]+l1*ovTol) \
                 and (N-1-bi[0])<=maxSpan and ((bull and d==1) or (bear and d==-1))
            if ok: return "待定"
    return None


NAMES = ["(1)","(2)","(3)","(4)","(5)","A","B","C","(1)'","(2)'"]
IMP   = {0,2,4,5,7,8}          # 推动型的腿：1 3 5 A C 以及下一轮的 1
P = dict(need5=True, ovTol=0.0, maxSpan=300, ret2=0.618, fib3=1.618, ret4=0.382,
         fib5=1.0, projRetr=0.5, retB=0.618, legs=4, horizon=480)

def scenario(z, prm=P, mintick=1e-8):
    pv, d, ext, ebi, N, close = z['pv'], z['d'], z['ext'], z['ebi'], z['N'], z['close']
    n = len(pv)
    if n < 2 or d == 0: return None
    # ── 1. 找能解释最多枢轴的那个位置：k = 当前在走第几浪（0..7 → 1,2,3,4,5,A,B,C）
    for k in range(min(7, n-1), -1, -1):
        Q = pv[n-1-k:]
        bull = not Q[0][2]; s = 1.0 if bull else -1.0
        if any(Q[i][2] != ((i % 2 == 1) == bull) for i in range(k+1)): continue
        if any(Q[i+1][0] <= Q[i][0] for i in range(k)): continue
        if k >= 1 and prm['maxSpan'] and (N-1-Q[0][0]) > prm['maxSpan']: continue
        x = [s*q[1] for q in Q]; xe = s*ext
        l1 = x[1]-x[0] if k >= 1 else None
        if k >= 2 and not x[2] > x[0]: continue                          # 浪2 不破起点
        if k >= 3 and not x[3] > x[1]: continue                          # 浪3 越过浪1
        if k >= 4 and not x[4] > x[1] - prm['ovTol']*l1: continue        # 浪4 不进浪1
        if k >= 5:
            l3, l5 = x[3]-x[2], x[5]-x[4]
            if l3 < l1 and l3 < l5: continue                             # 浪3 不最短
            if prm['need5'] and not x[5] > x[3]: continue                # 浪5 越过浪3
        if k >= 6 and not x[6] > x[0]: continue                          # A 不破推动浪起点
        if k >= 7 and not x[5] > x[7]: continue                          # B 不破推动浪终点
        # 正在走的那一浪，已经走出来的部分也不能违规
        if k == 1 and not xe > x[0]: continue
        if k == 3 and not xe > x[1] - prm['ovTol']*l1: continue
        if k in (5, 7) and not xe > x[0]: continue
        if k == 6 and not x[5] > xe: continue
        break
    else:
        return None
    # ── 2. 高度：每一浪的终点按斐波那契，再用硬规则夹紧
    E = [None]*10
    for j in range(k): E[j] = x[j+1]
    x0 = x[0]; eps = mintick; done = False; extended = False
    running = (N-1 - ebi) <= 1                 # 极值就在最近两根 → 这一浪还在走
    def target(j, ext):
        L1 = (E[0]-x0) if E[0] is not None else None
        if   j == 0: return xe
        elif j == 1: return max(E[0] - (0.786 if ext else prm['ret2'])*L1, x0 + eps)
        elif j == 2: return max(E[1] + (prm['fib3'] + (1.0 if ext else 0.0))*L1, E[0] + eps)
        elif j == 3: return max(E[2] - (0.5 if ext else prm['ret4'])*(E[2]-E[1]), E[0] - prm['ovTol']*L1 + eps)
        elif j == 4:
            l3 = E[2]-E[1]; t = E[3] + (prm['fib5'] + (0.618 if ext else 0.0))*L1
            if t <= E[2]:                      # 浪3 延长时「浪5=浪1」可能过不了浪3 → 换指引
                t = E[3] + 0.618*(E[2]-x0)
            if l3 < L1: t = min(t, E[3] + l3)
            if prm['need5']: t = max(t, E[2] + eps)
            return t
        elif j == 5:
            a = prm['projRetr']*(E[4]-x0) / (2 - prm['retB'])
            return max(E[4] - a*(1.618 if ext else 1.0), x0 + eps)
        elif j == 6: return min(E[5] + (0.786 if ext else prm['retB'])*(E[4]-E[5]), E[4] - eps)
        elif j == 7: return max(E[6] - (1.618 if ext else 1.0)*(E[4]-E[5]), x0 + eps)
        elif j == 8: return E[7] + L1
        elif j == 9: return E[8] - prm['ret2']*L1
    for j in range(k, 10):
        t = target(j, False)
        if j == k:
            up = (j % 2 == 0)
            beyond = (xe >= t) if up else (xe <= t)
            if beyond and running and j > 0:   # 越过目标、还在走 → 按延长浪取下一档
                t = target(j, True); extended = True
                beyond = (xe >= t) if up else (xe <= t)
            E[j] = xe if beyond else t
            done = beyond
        else:
            E[j] = t
    # ── 3. 角度：每条腿的耗时 = 价格长度 / 同类腿（推动 or 调整）在本组里的实测速度
    vi, vc = [], []
    for i in range(k):
        v = abs(x[i+1]-x[i]) / max(1, Q[i+1][0]-Q[i][0]); (vi if i in IMP else vc).append(v)
    if ebi - Q[k][0] >= 3:
        v = abs(xe-x[k]) / (ebi-Q[k][0]); (vi if k in IMP else vc).append(v)
    allv = [abs(pv[i+1][1]-pv[i][1]) / max(1, pv[i+1][0]-pv[i][0]) for i in range(max(0,n-11), n-1)]
    vAll = float(np.median(allv)) if allv else None
    if vAll is None: return None
    vImp = float(np.mean(vi + [vAll])); vCor = float(np.mean(vc + [vAll]))   # 本级别中位数算一票：收缩估计
    def dur(dp, j): return max(1, int(round(abs(dp) / (vImp if j in IMP else vCor))))
    # ── 4. 路径：从「现在」出发
    xc = s*close
    if done:
        f = k + 1; pts = [(ebi, E[k], k, True)]; tprev, xprev = ebi, E[k]
    else:
        f = k;     pts = [(N-1, xc, None, False)]; tprev, xprev = N-1, xc
    last = min(f + prm['legs'] - 1, 9)
    for j in range(f, last+1):
        t = tprev + dur(E[j]-xprev, j)
        if t <= N-1: t = N-1 + dur(E[j]-xc, j)       # 按速度本该走完却没走完 → 从现价重算剩余
        pts.append((t, E[j], j, False)); tprev, xprev = t, E[j]
    # 横向超出 TradingView 500 根未来上限时，等比例压缩（保持各腿角度的相对关系）
    span = pts[-1][0] - (N-1)
    if span > prm['horizon']:
        sc = prm['horizon']/span
        pts = [(t if t <= N-1 else (N-1) + max(1, int(round((t-(N-1))*sc))), xx, j, dn) for t,xx,j,dn in pts]
    return dict(k=k, bull=bull, s=s, done=done, extended=extended, p0=(Q[0][0], Q[0][1]), pts=[(t, s*xx, j, dn) for t,xx,j,dn in pts],
                vImp=vImp, vCor=vCor, E=[None if e is None else s*e for e in E], x0=s*x0, xc=close)

def check(res, N):
    """路径自检：时间递增、方向交替、硬规则"""
    pts = res['pts']; s = res['s']; errs = []
    ts = [p[0] for p in pts]
    if any(ts[i+1] <= ts[i] for i in range(len(ts)-1)): errs.append("时间不递增")
    if ts[-1] - (N-1) > 480: errs.append("超出 480 根")
    for a, b in zip(pts, pts[1:]):
        j = b[2]; up = (j % 2 == 0)
        if (s*(b[1]-a[1]) > 0) != up and abs(b[1]-a[1]) > 1e-9: errs.append(f"{NAMES[j]} 方向反了")
    E = [None if e is None else s*e for e in res['E']]; x0 = s*res['x0']
    if E[1] is not None and not E[1] > x0: errs.append("浪2 破起点")
    if E[3] is not None and E[0] is not None and not E[3] > E[0] - 1e-12: errs.append("浪4 进浪1")
    if E[7] is not None and not E[7] >= x0: errs.append("C 破推动浪起点")
    if E[6] is not None and not E[6] < E[4]: errs.append("B 破推动浪终点")
    return errs


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 108)
    print("A + B  覆盖率与路径自检    （同一组抽样点，旧版 vs 新版）")
    print("=" * 108)
    print(f"{'品种':<9}{'周期':<4} {'级别':<4}{'抽样':>5}{'旧版':>7}{'新版':>7}{'自检失败':>9}   新版选中的位置分布")
    tot = o_hit = n_hit = fail = 0
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        for tf in ("1w", "1d", "4h"):
            df = load(sym, tf); N0 = len(df)
            pts = sorted(rng.choice(np.arange(200 if tf != '1w' else 60, N0), size=min(60, N0 - 60), replace=False))
            for nm, m in zip(("细", "中", "粗"), AUTO[tf]):
                oh = nh = bad = 0; pos = {}
                for t in pts:
                    z = zz(df, m, upto=int(t))
                    if old_draws(z): oh += 1
                    r = scenario(z)
                    if r:
                        nh += 1
                        if check(r, z['N']): bad += 1
                        key = NAMES[r['k']] + ("?" if r['done'] else "")
                        pos[key] = pos.get(key, 0) + 1
                tot += len(pts); o_hit += oh; n_hit += nh; fail += bad
                dist = " ".join(f"{k}:{v}" for k, v in sorted(pos.items(), key=lambda kv: -kv[1])[:6])
                print(f"{sym:<9}{tf:<4} {nm:<4}{len(pts):>5}{oh/len(pts):>7.0%}{nh/len(pts):>7.0%}{bad:>9}   {dist}")
    print(f"\n合计 {tot} 个抽样点：旧版画得出 {o_hit/tot:.1%}，新版 {n_hit/tot:.1%}，新版路径自检失败 {fail}")

    print("\n" + "=" * 108)
    print("C  最新一根的推演路径")
    print("=" * 108)
    for sym, tf in (("BTCUSDT", "1w"), ("BTCUSDT", "1d"), ("BTCUSDT", "4h"), ("ETHUSDT", "1d")):
        df = load(sym, tf); seen = {}
        for nm, m in zip(("细", "中", "粗"), AUTO[tf]):
            z = zz(df, m); r = scenario(z)
            if not r:
                print(f"\n{sym} {tf} {nm}: 枢轴不足"); continue
            key = (r['k'], r['p0'])
            if key in seen:
                print(f"\n{sym} {tf} {nm}: 同「{seen[key]}」的场景，不重复画"); continue
            seen[key] = nm
            N = z['N']; step = df.index[-1] - df.index[-2]
            bt = lambda b: str(df.index[b] if b < N else df.index[-1] + (b - (N - 1)) * step)[:10]
            st = NAMES[r['k']] + ("? 越过目标·已回头" if r['done'] else (" 延长中" if r['extended'] else " 进行中"))
            print(f"\n{sym} {tf} {nm} ATR×{m}   现价 {r['xc']:,.2f}   {st}"
                  f"   速度 推动 {r['vImp']:,.2f} / 调整 {r['vCor']:,.2f} 每根")
            prev = None
            for t, px, j, dn in r['pts']:
                lab = "现在" if j is None else NAMES[j] + ("?" if dn else "")
                a = (f"   {px - prev[1]:+12,.2f} / {t - prev[0]:>3} 根 = {(px - prev[1]) / (t - prev[0]):+10,.2f}/根"
                     if prev else "")
                print(f"   {lab:<6} {bt(t)}  {px:>12,.2f}{a}"); prev = (t, px)


if __name__ == "__main__":
    main()
