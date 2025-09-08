# -*- coding: utf-8 -*-
# Discordボット：ETHシグナル（BB+RSI） + arb_bot出力転送 + ブラックリスト操作
# 依存: discord.py, ccxt, pandas, numpy, aiohttp, requests, tabulate

import os, re, json, asyncio, math, time
import discord
from discord import app_commands
import ccxt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Optional

# ======== 環境変数 ========
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # 必須: Discord Bot Token
CHANNEL1_ID = int(os.getenv("CHANNEL1_ID", "1414417853066514454"))  # ETHシグナル送信
CHANNEL2_ID = int(os.getenv("CHANNEL2_ID", "1413827349924806727"))  # arb出力転送
CHANNEL2_WEBHOOK_URL = os.getenv("CHANNEL2_WEBHOOK_URL", "")        # 任意: 送信をWebhookに切替

# ETHシグナル設定
SYMBOL           = os.getenv("SYMBOL", "ETH/USDT")
EXCHANGE_ID      = os.getenv("EXCHANGE_ID", "binance")
TIMEFRAME        = os.getenv("TIMEFRAME", "15m")
BB_PERIOD        = int(os.getenv("BB_PERIOD", "20"))
BB_STD           = float(os.getenv("BB_STD", "2.0"))
RSI_PERIOD       = int(os.getenv("RSI_PERIOD", "14"))
RSI_OB           = float(os.getenv("RSI_OB", "70.0"))  # Overbought → 下抜けで売り
RSI_OS           = float(os.getenv("RSI_OS", "30.0"))  # Oversold   → 上抜けで買い
CHECK_INTERVAL_S = int(os.getenv("CHECK_INTERVAL_S", "60"))

# arb_bot 実行設定
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
ARB_PATH   = os.getenv("ARB_PATH", "arb_bot.py")  # 同リポジトリに置く

# ブラックリスト永続化
BLACKLIST_FILE = os.getenv("BLACKLIST_FILE", "blacklist.json")

# ======== Discord Client ========
intents = discord.Intents.default()
intents.message_content = True  # コマンドの文字列引数用
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# ======== ブラックリスト =========
blacklist: Set[str] = set()

def load_blacklist():
    global blacklist
    p = Path(BLACKLIST_FILE)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                blacklist = {str(x).upper() for x in data}
        except Exception:
            blacklist = set()
    else:
        blacklist = set()

def save_blacklist():
    p = Path(BLACKLIST_FILE)
    p.write_text(json.dumps(sorted(blacklist), ensure_ascii=False, indent=2), encoding="utf-8")

load_blacklist()

# ======== ETH シグナル（BB + RSI）========
def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.fillna(0)

def bollinger(close: pd.Series, period: int, std_k: float):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = ma + std_k * std
    lower = ma - std_k * std
    return ma, upper, lower

touch_state = {"lower": False, "upper": False}

async def signal_task():
    await client.wait_until_ready()
    ch = client.get_channel(CHANNEL1_ID)
    if ch is None:
        print("[ERR] CHANNEL1_ID が見つからない。ボットの権限/配置を確認して。")
        return

    exchange_class = getattr(ccxt, EXCHANGE_ID)
    ex = exchange_class({"enableRateLimit": True})

    # 一度に十分な本数を取得
    need = max(BB_PERIOD, RSI_PERIOD) + 50

    while not client.is_closed():
        try:
            ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=need)
            if not ohlcv or len(ohlcv) < max(BB_PERIOD, RSI_PERIOD) + 5:
                await asyncio.sleep(CHECK_INTERVAL_S)
                continue

            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            close = df["c"].astype(float)
            _, bb_u, bb_l = bollinger(close, BB_PERIOD, BB_STD)
            r = rsi(close, RSI_PERIOD)

            # 最新確定足（-2）で判定
            i1, i2 = -2, -1

            # タッチ検知（確定足で）
            c_prev = close.iloc[i1]
            lb_prev = bb_l.iloc[i1]
            ub_prev = bb_u.iloc[i1]

            if not math.isnan(lb_prev) and c_prev <= lb_prev:
                touch_state["lower"] = True
                touch_state["upper"] = False

            if not math.isnan(ub_prev) and c_prev >= ub_prev:
                touch_state["upper"] = True
                touch_state["lower"] = False

            # RSI クロス判定
            r_prev = float(r.iloc[i1])
            r_now  = float(r.iloc[i2])

            if touch_state["lower"] and (r.iloc[i1] < RSI_OS) and (r.iloc[i2] >= RSI_OS):
                msg = f"買い時だ！ ({SYMBOL}, {TIMEFRAME})  下限→RSI{RSI_OS}上抜け"
                await ch.send(msg)
                touch_state["lower"] = False

            if touch_state["upper"] and (r.iloc[i1] > RSI_OB) and (r.iloc[i2] <= RSI_OB):
                msg = f"売り時だ！ ({SYMBOL}, {TIMEFRAME})  上限→RSI{RSI_OB}下抜け"
                await ch.send(msg)
                touch_state["upper"] = False

        except Exception as e:
            print("[signal_task ERROR]", e)

        await asyncio.sleep(CHECK_INTERVAL_S)

# ======== arb_bot 出力 → チャンネル2 ========
ALERT_RE = re.compile(r'^\[ALERT\]\s+([A-Z0-9]+)/([A-Z]+)\s+([0-9.]+)%')

async def post_channel2(content: str):
    if CHANNEL2_WEBHOOK_URL:
        # Webhookで送信（簡単・確実）
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(CHANNEL2_WEBHOOK_URL, json={"content": content}) as resp:
                if resp.status >= 300:
                    print("[Webhook Error]", resp.status, await resp.text())
    else:
        ch = client.get_channel(CHANNEL2_ID)
        if ch is not None:
            await ch.send(content)
        else:
            print("[ERR] CHANNEL2_ID が見つからない")

async def arb_task():
    await client.wait_until_ready()

    # arb_bot.py をサブプロセスで起動（あなたの設定をそのまま環境変数で渡す）
    env = os.environ.copy()
    # tekisuto.txtの推奨値はRenderの環境変数に設定しておく（ここではenvをそのまま利用）
    # 例: ENABLED_EXCHANGES, QUOTES, DEX_PAIRS, MIN_QVOL_*, MIN_DETECT_SPREAD_PCT など

    proc = await asyncio.create_subprocess_exec(
        PYTHON_BIN, ARB_PATH,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env
    )

    print("[arb_task] started")

    while True:
        line = await proc.stdout.readline()
        if not line:
            if proc.returncode is not None:
                print("[arb_task] process ended:", proc.returncode)
                break
            await asyncio.sleep(0.5)
            continue

        s = line.decode("utf-8", errors="ignore").rstrip()
        m = ALERT_RE.match(s)
        if m:
            base = m.group(1).upper()
            if base not in blacklist:
                # [ALERT] 行だけ転送。直後の 0x ルート行も拾いたいので軽く待つ
                await post_channel2(f"`{s}`")
                # 直後に続くルート行（インデント付き）を拾う
                try:
                    nxt = await asyncio.wait_for(proc.stdout.readline(), timeout=0.2)
                    ns = nxt.decode("utf-8", errors="ignore").rstrip()
                    if "[dex0x route]" in ns and base not in blacklist:
                        await post_channel2(f"`{ns}`")
                except Exception:
                    pass

# ======== Slash Commands: /blacklist ========
@tree.command(name="blacklist_show", description="ブラックリストを表示")
async def blacklist_show(interaction: discord.Interaction):
    items = sorted(blacklist)
    text = "（空）" if not items else ", ".join(items)
    await interaction.response.send_message(f"Blacklist: {text}", ephemeral=True)

@tree.command(name="blacklist_add", description="ブラックリストに銘柄を追加（例: BTC）")
@app_commands.describe(symbol="ベースシンボル（例: BTC, ETH, SOL）")
async def blacklist_add(interaction: discord.Interaction, symbol: str):
    s = symbol.strip().upper()
    if not s:
        await interaction.response.send_message("symbolが空", ephemeral=True)
        return
    blacklist.add(s)
    save_blacklist()
    await interaction.response.send_message(f"追加: {s}", ephemeral=True)

@tree.command(name="blacklist_remove", description="ブラックリストから削除")
@app_commands.describe(symbol="ベースシンボル（例: BTC, ETH, SOL）")
async def blacklist_remove(interaction: discord.Interaction, symbol: str):
    s = symbol.strip().upper()
    if s in blacklist:
        blacklist.remove(s)
        save_blacklist()
        await interaction.response.send_message(f"削除: {s}", ephemeral=True)
    else:
        await interaction.response.send_message(f"{s} は入ってない", ephemeral=True)

@tree.command(name="blacklist_clear", description="ブラックリストを全消し")
async def blacklist_clear(interaction: discord.Interaction):
    blacklist.clear()
    save_blacklist()
    await interaction.response.send_message("クリアした。", ephemeral=True)

# ======== 起動 ========
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    try:
        await tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print("[Slash Sync ERROR]", e)
    # 並行タスク
    asyncio.create_task(signal_task())
    asyncio.create_task(arb_task())

if not TOKEN:
    raise SystemExit("環境変数 DISCORD_BOT_TOKEN を設定してください。")
client.run(TOKEN)
