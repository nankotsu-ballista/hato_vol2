# -*- coding: utf-8 -*-
# Discordボット：ETHシグナル（BB+RSI） + arb_bot出力転送 + ブラックリスト操作
import os, re, json, asyncio, math
import discord
from discord import app_commands
import ccxt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Set

# ======== 環境変数 ========
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL1_ID = int(os.getenv("CHANNEL1_ID", "1414417853066514454"))
CHANNEL2_ID = int(os.getenv("CHANNEL2_ID", "1413827349924806727"))
CHANNEL2_WEBHOOK_URL = os.getenv("CHANNEL2_WEBHOOK_URL", "")

SYMBOL      = os.getenv("SYMBOL", "ETH/USDT")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
TIMEFRAME   = os.getenv("TIMEFRAME", "15m")
BB_PERIOD   = int(os.getenv("BB_PERIOD", "20"))
BB_STD      = float(os.getenv("BB_STD", "2.0"))
RSI_PERIOD  = int(os.getenv("RSI_PERIOD", "14"))
RSI_OB      = float(os.getenv("RSI_OB", "70.0"))
RSI_OS      = float(os.getenv("RSI_OS", "30.0"))
CHECK_INTERVAL_S = int(os.getenv("CHECK_INTERVAL_S", "60"))

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
ARB_PATH   = os.getenv("ARB_PATH", "arb_bot.py")

BLACKLIST_FILE = os.getenv("BLACKLIST_FILE", "blacklist.json")
GUILD_ID = int(os.getenv("GUILD_ID", "0")) or None

# ======== Discord Client ========
intents = discord.Intents.default()
intents.message_content = True
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
    Path(BLACKLIST_FILE).write_text(json.dumps(sorted(blacklist), ensure_ascii=False, indent=2), encoding="utf-8")
load_blacklist()

# ======== 指標 ========
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

async def fetch_ohlcv_async(ex, symbol, timeframe, limit):
    # ccxt は同期I/O。イベントループを塞がないようスレッドへ
    return await asyncio.to_thread(ex.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)

async def signal_task():
    await client.wait_until_ready()
    ch = client.get_channel(CHANNEL1_ID)
    if ch is None:
        print("[ERR] CHANNEL1_ID not found")
        return

    exchange_class = getattr(ccxt, EXCHANGE_ID)
    ex = exchange_class({"enableRateLimit": True})
    need = max(BB_PERIOD, RSI_PERIOD) + 50

    while not client.is_closed():
        try:
            ohlcv = await fetch_ohlcv_async(ex, SYMBOL, TIMEFRAME, need)
            if not ohlcv or len(ohlcv) < max(BB_PERIOD, RSI_PERIOD) + 5:
                await asyncio.sleep(CHECK_INTERVAL_S); continue

            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            close = df["c"].astype(float)
            _, bb_u, bb_l = bollinger(close, BB_PERIOD, BB_STD)
            r = rsi(close, RSI_PERIOD)

            i1, i2 = -2, -1
            c_prev = close.iloc[i1]
            lb_prev = bb_l.iloc[i1]
            ub_prev = bb_u.iloc[i1]

            if not math.isnan(lb_prev) and c_prev <= lb_prev:
                touch_state["lower"] = True; touch_state["upper"] = False
            if not math.isnan(ub_prev) and c_prev >= ub_prev:
                touch_state["upper"] = True; touch_state["lower"] = False

            if touch_state["lower"] and (r.iloc[i1] < RSI_OS) and (r.iloc[i2] >= RSI_OS):
                await ch.send(f"買い時だ！ ({SYMBOL}, {TIMEFRAME})  下限→RSI{RSI_OS}上抜け")
                touch_state["lower"] = False

            if touch_state["upper"] and (r.iloc[i1] > RSI_OB) and (r.iloc[i2] <= RSI_OB):
                await ch.send(f"売り時だ！ ({SYMBOL}, {TIMEFRAME})  上限→RSI{RSI_OB}下抜け")
                touch_state["upper"] = False

        except Exception as e:
            print("[signal_task ERROR]", e)

        await asyncio.sleep(CHECK_INTERVAL_S)

# ======== arb_bot 出力 → チャンネル2 ========
ALERT_RE = re.compile(r'^\[ALERT\]\s+([A-Z0-9]+)/([A-Z]+)\s+([0-9.]+)%')

async def post_channel2(content: str):
    if CHANNEL2_WEBHOOK_URL:
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
            print("[ERR] CHANNEL2_ID not found")

async def arb_task():
    await client.wait_until_ready()
    env = os.environ.copy()
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
                print("[arb_task] process ended:", proc.returncode); break
            await asyncio.sleep(0.5); continue
        s = line.decode("utf-8", errors="ignore").rstrip()
        m = ALERT_RE.match(s)
        if m:
            base = m.group(1).upper()
            if base not in blacklist:
                await post_channel2(f"`{s}`")
                try:
                    nxt = await asyncio.wait_for(proc.stdout.readline(), timeout=0.2)
                    ns = nxt.decode("utf-8", errors="ignore").rstrip()
                    if "[dex0x route]" in ns and base not in blacklist:
                        await post_channel2(f"`{ns}`")
                except Exception:
                    pass

# ======== Slash Commands ========
@tree.command(name="blacklist_show", description="ブラックリストを表示")
async def blacklist_show(interaction: discord.Interaction):
    items = sorted(blacklist)
    await interaction.response.send_message("Blacklist: " + ("（空）" if not items else ", ".join(items)), ephemeral=True)

@tree.command(name="blacklist_add", description="ブラックリストに銘柄を追加（例: BTC）")
@app_commands.describe(symbol="ベースシンボル（例: BTC, ETH, SOL）")
async def blacklist_add(interaction: discord.Interaction, symbol: str):
    s = symbol.strip().upper()
    if not s:
        await interaction.response.send_message("symbolが空", ephemeral=True); return
    blacklist.add(s); save_blacklist()
    await interaction.response.send_message(f"追加: {s}", ephemeral=True)

@tree.command(name="blacklist_remove", description="ブラックリストから削除")
@app_commands.describe(symbol="ベースシンボル（例: BTC, ETH, SOL）")
async def blacklist_remove(interaction: discord.Interaction, symbol: str):
    s = symbol.strip().upper()
    if s in blacklist:
        blacklist.remove(s); save_blacklist()
        await interaction.response.send_message(f"削除: {s}", ephemeral=True)
    else:
        await interaction.response.send_message(f"{s} は入ってない", ephemeral=True)

@tree.command(name="blacklist_clear", description="ブラックリストを全消し")
async def blacklist_clear(interaction: discord.Interaction):
    blacklist.clear(); save_blacklist()
    await interaction.response.send_message("クリアした。", ephemeral=True)

# ======== 起動 ========
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    try:
        if GUILD_ID:
            guild = discord.Object(id=GUILD_ID)
            tree.copy_global_to(guild=guild)
            await tree.sync(guild=guild)
            print(f"Slash commands synced to guild {GUILD_ID}.")
        else:
            await tree.sync()
            print("Slash commands synced globally.")
    except Exception as e:
        print("[Slash Sync ERROR]", e)
    asyncio.create_task(signal_task())
    asyncio.create_task(arb_task())

if not TOKEN:
    raise SystemExit("環境変数 DISCORD_BOT_TOKEN を設定しろ。")
client.run(TOKEN)
