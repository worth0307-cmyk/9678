# moomoo 版：combo_overlay + impulse_wave

把 TradingView 上的两个指标（`pine/combo_overlay.pine`、`pine/impulse_wave.pine`）搬到 moomoo，分两部分：

| | 做什么 | 在哪 |
|---|---|---|
| **A. 公式指标** | 在 moomoo 的 K 线图上画均线和斐波那契 | `formula/` 两个文件，粘贴进 moomoo 桌面版的指标编辑器 |
| **B. Python 策略** | 通过 OpenD 读你的自选股分组和日 K，算出两个指标的全部内容，回测 8 套规则，每天出一张筛选表，还能把符合条件的股票写回 moomoo 的一个分组 | `run.py` |

为什么要拆开：moomoo 的公式语言和通达信差不多，**没有循环、数组，也不能引用自己上一根的值**。
SuperTrend 的轨道要逐根「只升不降」，数浪要记住一串转折点，这两样都写不进公式。
moomoo 的选股器同样只能用它内置的指标（MA、EMA、RSI、KDJ、MACD、BOLL 和 K 线形态），不能放自定义公式。
所以公式里只放画得出来的部分，其余的由 Python 算好，再把结果（股票名单）写回 moomoo。

| TradingView 里的东西 | moomoo 公式 | Python |
|---|---|---|
| 均线带 MA 20/50/100、EMA 20/50/100 | ✓ `combo_ma.txt` | 不进规则 |
| 200 日均线 | ✓ | ✓ 规则 S0、S4 |
| 50 周均线 | ≈ 250 日均线（只在日线图上对） | ✓ 精确值，显示在筛选表 |
| 斐波那契窗口 | ✓ `fib_1y.txt`（一年 = 250 根） | ✓ 筛选表里的「一年区间位置」 |
| SuperTrend（ATR 10，倍数 3） | ✗ 写不出来 | ✓ 规则 S1 |
| 推动浪数浪、失效价、D/W 趋势小标签、1-5 确认 | ✗ 写不出来 | ✓ 规则 S2、S3 |
| 背离（11 个指标的枢轴比较） | ✗ | ✗ 没移植，只是看图用的 |
| 趋势小标签里的 H（12 小时） | — | ✗ moomoo 的股票没有 12 小时 K 线，只用 D、W |

---

## 先想好在哪台电脑上跑

三样东西，别搞混：

| 程序 | 干什么 | 装在哪 |
|---|---|---|
| **moomoo 桌面版**（`moomoo_desktop_xxx.exe`） | 平常看盘、下单；**公式指标要贴在这里** | 你的 Windows 电脑 |
| **moomoo OpenD** | 给程序用的行情网关，**和桌面版是两个不同的程序**，要单独下载 | **和 Python 程序同一台电脑** |
| Python 程序 `run.py` | 回测、筛选 | 同上 |

Python 只连 `127.0.0.1`（本机）上的 OpenD，**OpenD 开在 Windows、程序在 VPS 上是连不到的**。
推荐先**全部放在 Windows 电脑上**跑通：OpenD 有图形界面，登录、短信验证码都在窗口里点。
以后想让 VPS 每天自动出筛选表，再在 VPS 上装命令行版 OpenD（要改配置文件、用 telnet 输验证码，麻烦一些）。

## 第一步：安装 OpenD 并登录

OpenD 是 moomoo 官方的本地网关程序。Python 不直接连 moomoo 服务器，而是连你电脑上的 OpenD，
登录和行情权限都在 OpenD 里处理。

1. **下载**：在 moomoo 官网的下载页找「OpenAPI」，下载 **moomoo OpenD**，选带界面的 GUI 版
   （Windows / macOS 都有）。官方文档：<https://openapi.moomoo.com/moomoo-api-doc/>
2. **安装并打开**，用你的 moomoo 账号（moomoo ID / 手机 / 邮箱）和密码登录。
   第一次用 API 可能要做一份问卷、同意 API 使用协议，照提示做完。
3. 登录成功后 OpenD 窗口会显示监听地址，默认是 **127.0.0.1:11111**，保持它开着。
   改过端口的话，下面的命令都加上 `--port 你的端口`。
4. OpenD 和 moomoo App 可以同时登录同一个账号，互不影响。

> 我这边的环境连不上 moomoo 的服务器，OpenD 这一段没法替你实测。
> 哪一步卡住了，把 OpenD 窗口或者命令行的报错截图发我。

## 第二步：拿到代码，装 Python 依赖

1. 代码在 GitHub 仓库 `worth0307-cmyk/9678` 的 `claude/btcusdt-trading-system-hl6119` 分支。
   网页上切到这个分支 → Code → Download ZIP 解压，或者
   `git clone -b claude/btcusdt-trading-system-hl6119 https://github.com/worth0307-cmyk/9678.git`。
2. 装 Python 3.9 或更新的版本。

**Windows**：从 python.org 下载安装，安装第一页勾上 *Add python.exe to PATH*。
然后打开 PowerShell，把下面五行整段粘贴进去：下载代码到你的用户目录、解压、装依赖。
以后代码有更新，再粘一遍就是更新（会覆盖旧文件，`cache/`、`out/` 里的东西不受影响）。

```
cd $HOME; $ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest "https://github.com/worth0307-cmyk/9678/archive/refs/heads/claude/btcusdt-trading-system-hl6119.zip" -OutFile 9678.zip
Expand-Archive 9678.zip -DestinationPath . -Force
cd 9678-claude-btcusdt-trading-system-hl6119
python -m pip install -r moomoo_strat/requirements.txt
```

之后每次打开 PowerShell，先 `cd $HOME\9678-claude-btcusdt-trading-system-hl6119` 进到这个文件夹，再跑下面的命令。

**Linux / VPS**（Ubuntu 24.04 这类新系统）：系统自带的 Python 不让直接 `pip install`
（报 `externally-managed-environment`），要先建一个虚拟环境；命令也是 `python3` 不是 `python`。
VPS 上仓库在 `~/9678`、虚拟环境在 `~/9678/.venv`（`scripts/vps_setup.sh` 建的）：

```
cd ~/9678 && git pull
.venv/bin/pip install -r moomoo_strat/requirements.txt
.venv/bin/python moomoo_strat/run.py check
```

没有 `.venv` 的话先 `python3 -m venv .venv`（报错就先 `apt install -y python3-venv`）。
下面的命令在 VPS 上都把开头的 `python` 换成 `.venv/bin/python`。

### Windows 最省事：双击 `moomoo_tool.bat`

**把 `moomoo_strat/moomoo_tool.bat` 复制到桌面**双击运行；第一次会自动下载代码、装依赖（只需要事先装好 Python）。
之后出来一个中文菜单，输编号回车：检查连接、回测一个分组、回测 79 只大盘股、今日筛选、同步五浪信号分组、
打开结果文件夹、更新代码。不用开 PowerShell，不用 cd。

- 选分组时会列出你所有的自选股分组，按编号选，不用打中文；直接回车 = 上次用的那个。
- bat 文件本身只有英文：cmd 读含中文的 UTF-8 bat 会把行读错位（中文菜单行被当成命令执行），
  所以中文菜单放在 Python 里（`run.py menu`）。

## 第三步：检查连接

```
python moomoo_strat/run.py check
```

正常会打印：OpenD 版本、行情是否已登录、历史 K 线额度、你所有的自选股分组名。
报「连不上 OpenD」就是这台电脑上的 OpenD 没开、没登录，或者端口不对。
分组名要和这里打印的**一字不差**（系统分组在不同语言版本里可能叫「全部」或 "All"）。
想顺便看某个分组里有哪些股票：

```
python moomoo_strat/run.py check --group 美股
```

**历史 K 线额度**：每 30 天内能拉的**不同股票**只数有上限，按账户资产和交易量分档，`check` 会打印剩余多少。
同一只股票 30 天内重复拉不再扣额度。拉过的数据当天会缓存在 `moomoo_strat/cache/`，当天重跑不再请求。

如果连上了，但拉 K 线时报「无权限」之类的错误，说明这个市场的 API 行情权限没开，
看官方文档里「行情权限」那一节，或者在 moomoo App 里搜「OpenAPI」。

## 第四步：回测

```
python moomoo_strat/run.py backtest --group 美股
```

`美股` 换成你在 `check` 里看到的分组名。默认从 2015-01-01 拉前复权日 K。几十只股票第一次拉大约几分钟（每 30 秒最多 60 次请求，程序自动放慢）。
跑完在终端打印报告，同时存到 `moomoo_strat/out/`：

- `backtest_<分组>_<日期>.md`：总表、零假设、前后半段、五浪逐笔、逐只汇总
- `_rules.csv` / `_stocks.csv` / `_trades.csv`：同样的数字，方便用 Excel 打开
- `_equity.png`：各规则的净值曲线

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--start` | 2015-01-01 | 从哪天开始 |
| `--cost-bps` | 5 | 每边费用（滑点 + 平台费），单位 bp。**港股有印花税，建议 15** |
| `--codes` | | 不用分组时直接给代码：`--codes US.AAPL,US.MSFT,HK.00700` |
| `--codes-file` | | 代码清单文件，一行一个。`universe/us_large80.txt` 是现成的 80 只美股大盘股（见下面「样本外检验」） |
| `--refresh` | | 忽略今天的缓存，重新拉 |
| `--st` | 10,3 | SuperTrend 的 ATR 周期和倍数，和 TradingView 图上的设置一致才对得上（screen 也认） |
| `--null-reps` | 300 | 随机择时零假设抽多少次 |

### 8 套规则

全部**只做多**，每只股票固定分 1/N 的资金，第 t 天收盘算信号，**第 t+1 天开盘成交**。

| 规则 | 什么时候持有 | 来自 |
|---|---|---|
| **S0** 基准 | 收盘在 200 日均线上方 | 最朴素的趋势过滤，当尺子用：复杂的规则要是连它都比不过，就没必要用 |
| **S1** | SuperTrend 是多头（绿线） | combo_overlay 第 3 节，默认 ATR 10、倍数 3。你图上改过参数的话加 `--st 周期,倍数`，比如 `--st 12,2.5` |
| **S2** | 推动浪趋势 **D⬆ 且 W⬆** | impulse_wave 的 H/D/W 小标签（中档），股票只有 D、W |
| S2D | 只看 D⬆ | 诊断：S2 的两半各自有没有用 |
| S2W | 只看 W⬆ | 同上 |
| **S3** | 一组**向下**的 1-5 刚确认（图上的三角） → 次日开盘买入；碰到调整目标线（回撤五浪全长的 50%）就卖；收盘跌破浪5 终点、或同级别又确认了一组 → 次日开盘卖 | impulse_wave 的「五浪走完做调整」，日线「细」档（ATR×2） |
| S3M | 同上，「中」档（ATR×4） | |
| **S4** | S1 **且** S2 **且** 收盘在 200 日均线上方 | 组合 |

反方向的「涨完五浪做空」也会算，但只出逐笔统计，不进组合（多数人账户做空不方便）。

### 怎么读结论

只做多的规则有一半时间空仓，回撤自然比满仓的买入持有小 —— 这不说明规则有本事，把仓位砍一半也能做到。
所以每套规则都和**同样仓位的买入持有**比：

- **Sharpe 差** > 0 才算择时带来了东西。
- **p 自助法**、**p 随机择时**：这个差有多大可能只是运气。
  随机择时 = 把持仓序列整体平移一个随机天数，仓位比例、每段拿多久、换手都不变，只是时机错开。
- **去偏**：一共试了 8 套，挑最好的那套天然占便宜，这是扣掉这层之后「真有超额」的概率。
- **前/后半段**两段的 Sharpe 差方向一致才像真的。

报告最后一列直接给结论：「显著好于同仓位买入持有」要求三个 p 都过关、去偏 > 0.9，
否则是「偏好但证据不够」「略好、不显著」或「不如同仓位买入持有」。

**五浪规则（S3、S3M）另外看。** 它们平均仓位只有百分之一二，大部分时间空仓，和买入持有比 Sharpe 没有意义，
结论按**单笔**下，和两个零假设比：

- **随机括号**：同方向、同目标和止损距离、同持有上限，随便哪天进场。
- **整体平移**：全部交易一起挪同一个随机天数。自选股里的股票常常同涨同跌，一次大跌会让好几只同一周一起出信号；
  随机括号把它们当成好几次独立的机会，p 值偏乐观，整体平移保留这种扎堆，更严。
- **独立时段**：进场日期相隔不到两周的算同一次。18 笔如果只来自 6 次大跌，能下的结论只有 6 次那么多。

要两个 p 都 < 0.05、而且独立时段 ≥ 20，才判「单笔显著好于随机进场」。

### 样本外检验

在自选股上跑出来「有点像有用」的规则，要换一批**没参与挑选**的股票再测一次，参数一个不改，事先定好怎么算过关。
`universe/us_large80.txt` 是 80 只 2015 年前就上市的美股大盘股，不含 7姐妹 和 SPY / QQQ，
各行业都有，涨得好的和落后的都有：

```
python moomoo_strat/run.py backtest --codes-file moomoo_strat/universe/us_large80.txt
```

会用掉 80 只的历史 K 线额度（30 天滚动）；额度不够时程序会在拉数据之前停下来，不会拉一半。

**两个提醒**：

1. **自选股有幸存者偏差。** 分组里的股票是你现在挑出来的，它们过去多半涨得不错（所以你才关注），
   买入持有和规则的绝对收益都会偏高。「规则 vs 同仓位买入持有」两边都吃这个偏差，比较本身受影响小一些，
   但别把绝对收益当成未来能拿到的数。
2. **别从逐只明细里挑「这只股票上有效」的规则。** 几十只里总有几只碰巧对得上，那是挑出来的，不是规则的本事。

## 第五步：每日筛选

收盘后（或第二天开盘前）跑：

```
python moomoo_strat/run.py screen --group 美股
```

每只股票一行：

| 列 | 意思 |
|---|---|
| D / W | 推动浪趋势箭头（中档），规则用的 W 是**上一个走完的周** |
| W本周至今 | TradingView 小标签实时显示的那个，周五收盘才定 |
| ST / ST线(止损) / ST已持续 | SuperTrend 方向、那条线的价格（多头时就是止损位）、翻转后过了几天 |
| D位置 / W位置 | 数到第几浪，比如「浪3进行中」「浪5走完→A」 |
| D失效价 / W失效价 | 价格越过这里，这套数浪就违反艾略特硬规则、作废（图上的「失」） |
| 200日线 / 50周线 / 一年区间位置 | 一年区间位置 = 现价在一年最高最低之间的百分位，0% 在低点、100% 在高点 |
| S0 S1 S2 S4 | ✓ = 这条规则现在持有 |
| S3 | ✓ = 今天收盘刚确认「跌完五浪」（明天开盘买），或者五浪多单还拿着 |
| 今日变化 | 今天哪条规则进场 / 离场 |
| 五浪 | 五浪做多的细节：明天开盘进场、目标价、收盘跌破哪里就卖；或者正在持有的那一笔 |

五浪只出**做多**：「涨完五浪做空」在 7姐妹 和 79 只大盘股上都比随机进场还差。

结果同时存成 `out/screen_<分组>_<日期>.csv`。

**盘中跑会用到当天还没走完的 K 线**，信号可能到收盘又变了，所以请收盘后再跑。

### 把信号写回 moomoo

先在 moomoo App 里新建一个空的自选股分组，比如叫「策略信号」，然后：

```
python moomoo_strat/run.py screen --group 美股 --to-group 策略信号 --confirm
```

程序会把「策略信号」这个分组**同步**成 S3 当前的股票（今天刚出五浪做多信号、或者五浪多单还拿着）：
新符合的加进去，不再符合的移出去。`--confirm` 会先列出要加入、移出哪些，等你输 y 才改（菜单里的第 5 项默认就这样）。

- **它只动你在 `--to-group` 里指定的那一个分组**，不会碰别的分组。
- **系统分组（全部、美股、港股……）一律拒绝**：同步会把不在信号里的股票都移出去，等于清空你的自选。
  请专门建一个空的自定义分组给它用，别填「7姐妹」这种你自己在用的分组。
- `--rule` 默认 S3，也可以选 S0 / S1 / S2 / S4，不过回测里这四条都不比同仓位拿着好。

## 公式指标：在 moomoo 图上画均线和斐波那契

需要 moomoo **桌面版**（手机 App 不能编辑公式）：打开任意一只股票的 K 线图 → 指标 → 自定义指标（指标编辑器）→ 新建，
类型选**主图叠加**，把文件内容整段粘贴进去，保存后在主图指标里勾上。

- `formula/combo_ma.txt`：MA 20/50/100、EMA 20/50/100、200 日均线、50 周均线。
  50 周均线用 250 日均线近似，**只在日线图上对**；在周线图上请把最后一行的 250 改成 50。
- `formula/fib_1y.txt`：一年窗口（250 根日 K）的斐波那契 0、0.236、0.382、0.5、0.618、0.786、1。
  要「半年」窗口就把两处 250 改成 125。

颜色和线宽在 moomoo 的指标设置里调。

> 我没法在 moomoo 里实测这两个公式。语法按 moomoo / 通达信的通用写法：`名字:公式;` 是画出来的线，
> `名字:=公式;` 是不画的中间变量。粘贴后如果报错，把报错截图发我，我按 moomoo 的提示改。

**和 TradingView 的一处差别**：TradingView 版 combo_overlay 的斐波那契「1 年」是按日历天数换算根数的
（365 天 → 365 根），在加密货币上正好一年，但股票一年只有约 252 根交易日，所以它在股票日线上其实是约 17 个月。
这里的公式和 Python 都按 250 根 = 一年。

## 不连 OpenD 也能试

用任意日 K 的 CSV 目录（一个文件一只，列名认 `time_key/date/日期`、`open/high/low/close` 或中文的开盘最高最低收盘）：

```
python moomoo_strat/run.py backtest --csv-dir data --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,TAOUSDT,HYPEUSDT
python moomoo_strat/run.py screen   --csv-dir data --symbols BTCUSDT,ETHUSDT
```

上面这条用的是仓库里六个币的数据，只是为了验证程序能跑通，**不代表股票上的结果**。
（顺带一提：六个币上 8 套规则没有一套显著好于同仓位的买入持有；五浪「做空」那一侧在日线「细」档 10 笔里赢 9 笔，
和 `REPORT_IMPULSE_WAVE.md` 的结论一样，笔数太少，靠不住。）

## 文件

```
run.py         命令行：check / backtest / screen
data.py        OpenD：连接、自选股分组、前复权日 K（自动翻页、限速、当天缓存）；离线 CSV
combo.py       SuperTrend、50 周均线、一年区间位置
impulse.py     推动浪：ZigZag、1-5 确认、当前位置、失效价、D/W 趋势
strategy.py    8 套规则、次日开盘成交、五浪交易、统计
report.py      回测报告、零假设、净值图、每日筛选
formula/       moomoo 公式指标
```

`tests/test_moomoo_strat.py` 检查三件事：推动浪部分和 `scripts/60`（又和 `scripts/59`、pine 对过）逐根一致；
把未来的数据砍掉，过去每一根的指标和持仓都不变（没有未来函数）；成交口径按手算的例子对得上。
