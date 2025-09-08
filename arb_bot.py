# -*- coding: utf-8 -*-
# Arbitrage Scanner: CEX + JP Exchanges + DEX(0x)
# 依存: pip install requests tabulate

import os, time, sys, statistics
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import requests
from tabulate import tabulate

# ====== 設定（環境変数で上書き可能） ======
ENABLED_EXCHANGES = [s.strip().lower() for s in os.getenv(
    "ENABLED_EXCHANGES", "bybit,mexc,bitget,bitflyer,bitbank,coincheck,gmocoin,dex0x"
).split(",") if s.strip()]

QUOTES = {s.strip().upper() for s in os.getenv("QUOTES", "USDT,USDC,USD,JPY").split(",")}
INTERVAL_S = int(os.getenv("INTERVAL_S", "10"))

# アラート（ビープ等）用の別しきい値：実効 = base + fee
ALERT_BASE_PCT   = float(os.getenv("ALERT_PCT", "0.80"))  # ←例: 0.80%
FEE_BUFFER_PCT   = float(os.getenv("FEE_BUFFER_PCT", "0.20"))
ALERT_EFF_PCT    = ALERT_BASE_PCT + FEE_BUFFER_PCT        # ←例: 1.00%
TOPN             = int(os.getenv("TOPN", "200"))

# 検出フィルタ：この%未満は“検出しない”（出力に載せない）
MIN_DETECT_SPREAD_PCT = float(os.getenv("MIN_DETECT_SPREAD_PCT", "0.8"))

# ノイズ・安全フィルタ
MIN_EXCH_FOR_PAIR    = int(os.getenv("MIN_EXCH_FOR_PAIR","2"))
OUTLIER_TOL_PCT      = float(os.getenv("OUTLIER_TOL_PCT","30.0"))
MAX_SPREAD_CAP_PCT   = float(os.getenv("MAX_SPREAD_CAP_PCT","10.0"))
REQUIRE_RATIO_MAX    = float(os.getenv("REQUIRE_PRICE_RATIO_MAX","5.0"))
MIN_BASE_LEN         = int(os.getenv("MIN_BASE_LEN","2"))
BLACKLIST_SYMBOLS    = {s.strip().upper() for s in os.getenv("BLACKLIST_SYMBOLS","NEIRO,RED").split(",") if s.strip()}

# 出来高フィルタ（クオート建て / 24h）
MIN_QVOL = {
    "USDT": float(os.getenv("MIN_QVOL_USDT", "300000")),
    "USDC": float(os.getenv("MIN_QVOL_USDC", os.getenv("MIN_QVOL_USDT", "300000"))),
    "USD":  float(os.getenv("MIN_QVOL_USD",  "200000")),
    "JPY":  float(os.getenv("MIN_QVOL_JPY",  "30000000")),
}
REQUIRE_QVOL_BOTH_SIDES = os.getenv("REQUIRE_QVOL_BOTH_SIDES", "0") == "1"  # 片側OKをデフォ（DEX考慮）
ALLOW_UNKNOWN_QVOL      = os.getenv("ALLOW_UNKNOWN_QVOL", "1") == "1"       # DEXはUnknownが多い→許容

# DEXのルート表示（0/1）
SHOW_DEX_ROUTE = os.getenv("SHOW_DEX_ROUTE", "1") == "1"

SESSION = requests.Session()
TIMEOUT = (7, 12)

# DEXのルーティング情報を格納（(base,quote) -> str）
ROUTES: Dict[Tuple[str,str], str] = {}

# ====== ユーティリティ ======
def parse_f(x) -> float:
    try: return float(x)
    except: return 0.0

def split_base_quote(sym: str, allowed: set) -> Tuple[str,str]:
    s = sym.upper().replace("-", "").replace("_", "").replace("/", "")
    for q in sorted(allowed, key=len, reverse=True):
        if s.endswith(q):
            base = s[:-len(q)]
            if base:
                return base, q
    return "", ""

def beep():
    try:
        import winsound
        winsound.Beep(1200, 220)
    except Exception:
        sys.stdout.write('\a'); sys.stdout.flush()

# ====== アダプタ基底 ======
# fetch() は {(base, quote): (bid, ask, qvol_quote)} を返す
class Adapter:
    name = "base"
    def fetch(self) -> Dict[Tuple[str,str], Tuple[float,float, Optional[float]]]:
        raise NotImplementedError

# ====== CEX ======
class Bybit(Adapter):
    name = "bybit"
    def fetch(self):
        url = "https://api.bybit.com/v5/market/tickers?category=spot"
        r = SESSION.get(url, timeout=TIMEOUT); r.raise_for_status()
        data = (r.json() or {}).get("result", {}).get("list", []) or []
        out = {}
        for it in data:
            sym = it.get("symbol","")
            base, quote = split_base_quote(sym, QUOTES)
            if not base or len(base)<MIN_BASE_LEN or base in BLACKLIST_SYMBOLS: continue
            bid = parse_f(it.get("bid1Price")); ask = parse_f(it.get("ask1Price"))
            qvol = parse_f(it.get("turnover24h"))
            if bid>0 and ask>0: out[(base,quote)] = (bid, ask, qvol if qvol>0 else None)
        return out

class MEXC(Adapter):
    name = "mexc"
    _qmap = None
    def _ensure_qvol(self):
        if MEXC._qmap is not None: return
        try:
            rr = SESSION.get("https://api.mexc.com/api/v3/ticker/24hr", timeout=TIMEOUT)
            if rr.status_code==200:
                arr = rr.json()
                if isinstance(arr, list):
                    MEXC._qmap = {str(x.get("symbol","")): parse_f(x.get("quoteVolume")) for x in arr}
        except Exception:
            MEXC._qmap = {}
    def fetch(self):
        self._ensure_qvol()
        r = SESSION.get("https://api.mexc.com/api/v3/ticker/bookTicker", timeout=TIMEOUT); r.raise_for_status()
        rows = r.json(); rows = rows if isinstance(rows, list) else [rows]
        out={}
        for it in rows:
            sym = it.get("symbol",""); base, quote = split_base_quote(sym, QUOTES)
            if not base or len(base)<MIN_BASE_LEN or base in BLACKLIST_SYMBOLS: continue
            bid = parse_f(it.get("bidPrice")); ask = parse_f(it.get("askPrice"))
            qvol = None
            if isinstance(MEXC._qmap, dict): qvol = MEXC._qmap.get(sym) or None
            if bid>0 and ask>0: out[(base,quote)] = (bid, ask, qvol)
        return out

class BingX(Adapter):
    name = "bingx"
    _qmap = None
    def _ensure_qvol(self):
        if BingX._qmap is not None: return
        try:
            rr = SESSION.get("https://open-api.bingx.com/openApi/spot/v1/ticker/24hr", timeout=TIMEOUT)
            if rr.status_code==200:
                j = rr.json() or {}; data = j.get("data", [])
                if isinstance(data, list):
                    BingX._qmap = {str(x.get("symbol","")): parse_f(x.get("quoteVolume")) for x in data}
        except Exception:
            BingX._qmap = {}
    def fetch(self):
        self._ensure_qvol()
        r = SESSION.get("https://open-api.bingx.com/openApi/spot/v1/ticker/bookTicker", timeout=TIMEOUT); r.raise_for_status()
        data = (r.json() or {}).get("data", []) or []
        out={}
        for it in data:
            sym = it.get("symbol",""); base, quote = split_base_quote(sym, QUOTES)
            if not base or len(base)<MIN_BASE_LEN or base in BLACKLIST_SYMBOLS: continue
            bid = parse_f(it.get("bidPrice")); ask = parse_f(it.get("askPrice"))
            qvol = None
            if isinstance(BingX._qmap, dict): qvol = BingX._qmap.get(sym) or None
            if bid>0 and ask>0: out[(base,quote)] = (bid, ask, qvol)
        return out

class Bitget(Adapter):
    name = "bitget"
    def fetch(self):
        r = SESSION.get("https://api.bitget.com/api/spot/v1/market/tickers", timeout=TIMEOUT); r.raise_for_status()
        data = r.json().get("data",[]) or []
        out={}
        for it in data:
            sym = it.get("symbol",""); base, quote = split_base_quote(sym, QUOTES)
            if not base or len(base)<MIN_BASE_LEN or base in BLACKLIST_SYMBOLS: continue
            bid = parse_f(it.get("buyOne") or it.get("bestBid")); ask = parse_f(it.get("sellOne") or it.get("bestAsk"))
            qvol = parse_f(it.get("quoteVolume") or it.get("usdtVolume"))
            if bid>0 and ask>0: out[(base,quote)] = (bid, ask, qvol if qvol>0 else None)
        return out

# ====== 日本の取引所 ======
class Bitflyer(Adapter):
    name = "bitflyer"
    def _mkts(self) -> List[str]:
        r = SESSION.get("https://api.bitflyer.com/v1/markets", timeout=TIMEOUT); r.raise_for_status()
        out=[]
        for m in r.json():
            pc = m.get("product_code",""); mtype=(m.get("market_type") or "").lower()
            if pc.endswith("_JPY") and mtype=="spot": out.append(pc)
        return sorted(set(out))
    def fetch(self):
        out={}
        for pc in self._mkts():
            try:
                r = SESSION.get("https://api.bitflyer.com/v1/ticker", params={"product_code": pc}, timeout=TIMEOUT)
                if r.status_code!=200: continue
                j=r.json()
                bid=parse_f(j.get("best_bid")); ask=parse_f(j.get("best_ask"))
                ltp=parse_f(j.get("ltp"))
                vbp=parse_f(j.get("volume_by_product"))
                mid=(bid+ask)/2 if (bid>0 and ask>0) else (ltp if ltp>0 else 0)
                qvol = vbp*mid if (vbp>0 and mid>0) else None
                base, quote = split_base_quote(pc, QUOTES)
                if base and bid>0 and ask>0 and len(base)>=MIN_BASE_LEN and base not in BLACKLIST_SYMBOLS:
                    out[(base,quote)] = (bid, ask, qvol)
            except Exception: pass
        return out

class Bitbank(Adapter):
    name = "bitbank"
    DEFAULT_PAIRS = [
        "btc_jpy","eth_jpy","xrp_jpy","sol_jpy","ltc_jpy","bch_jpy","ada_jpy","dot_jpy",
        "link_jpy","xlm_jpy","xtz_jpy","mona_jpy","matic_jpy","avax_jpy","doge_jpy",
        "shib_jpy","sand_jpy","atom_jpy","grt_jpy","mana_jpy","enj_jpy","chz_jpy","iost_jpy","imx_jpy"
    ]
    def fetch(self):
        out={}
        pairs = [s.strip().lower() for s in os.getenv("BITBANK_PAIRS","").split(",") if s.strip()] or self.DEFAULT_PAIRS
        for p in pairs:
            url=f"https://public.bitbank.cc/{p}/ticker"
            try:
                r=SESSION.get(url, timeout=TIMEOUT)
                if r.status_code!=200: continue
                d=(r.json() or {}).get("data",{}) or {}
                bid=parse_f(d.get("buy")); ask=parse_f(d.get("sell")); last=parse_f(d.get("last"))
                vol=parse_f(d.get("vol"))
                mid=(bid+ask)/2 if (bid>0 and ask>0) else (last if last>0 else 0)
                qvol = vol*mid if (vol>0 and mid>0) else None
                base, quote = split_base_quote(p.upper(), QUOTES)
                if base and bid>0 and ask>0 and len(base)>=MIN_BASE_LEN and base not in BLACKLIST_SYMBOLS:
                    out[(base,quote)] = (bid, ask, qvol)
            except Exception: pass
        return out

class Coincheck(Adapter):
    name = "coincheck"
    DEFAULT_PAIRS = ["btc_jpy","eth_jpy","xrp_jpy","sol_jpy","ltc_jpy","bch_jpy"]
    def fetch(self):
        out={}
        pairs = [s.strip().lower() for s in os.getenv("COINCHECK_PAIRS","").split(",") if s.strip()] or self.DEFAULT_PAIRS
        base_url="https://coincheck.com/api/ticker"
        for p in pairs:
            try:
                r = SESSION.get(base_url, params={"pair": p}, timeout=TIMEOUT)
                if r.status_code!=200 and p=="btc_jpy":
                    r = SESSION.get(base_url, timeout=TIMEOUT)  # BTC/JPYフォールバック
                if r.status_code!=200: continue
                j=r.json() or {}
                bid=parse_f(j.get("bid")); ask=parse_f(j.get("ask")); last=parse_f(j.get("last"))
                vol=parse_f(j.get("volume"))
                mid=(bid+ask)/2 if (bid>0 and ask>0) else (last if last>0 else 0)
                qvol=vol*mid if (vol>0 and mid>0) else None
                base, quote = split_base_quote(p.upper(), QUOTES)
                if base and bid>0 and ask>0 and len(base)>=MIN_BASE_LEN and base not in BLACKLIST_SYMBOLS:
                    out[(base,quote)] = (bid, ask, qvol)
            except Exception: pass
        return out

class GMOCoin(Adapter):
    name = "gmocoin"
    def fetch(self):
        r=SESSION.get("https://api.coin.z.com/public/v1/ticker", timeout=TIMEOUT); r.raise_for_status()
        data=(r.json() or {}).get("data",[]) or []
        out={}
        for it in data:
            sym=(it.get("symbol") or "").upper()
            if not sym: continue
            base, quote = sym, "JPY"
            if quote not in QUOTES: continue
            if len(base)<MIN_BASE_LEN or base in BLACKLIST_SYMBOLS: continue
            bid=parse_f(it.get("bid")); ask=parse_f(it.get("ask")); last=parse_f(it.get("last"))
            vol=parse_f(it.get("volume"))
            mid=(bid+ask)/2 if (bid>0 and ask>0) else (last if last>0 else 0)
            qvol=vol*mid if (vol>0 and mid>0) else None
            if bid>0 and ask>0: out[(base,quote)] = (bid, ask, qvol)
        return out

# ====== DEX (0x Aggregator) ======
TOKENS = {
    "WETH": {"address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "decimals": 18},
    "ETH":  {"address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "decimals": 18},  # alias to WETH
    "WBTC": {"address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "decimals": 8},
    "USDT": {"address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "decimals": 6},
    "USDC": {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "decimals": 6},
}
def dex_notional_for_quote(quote: str) -> float:
    quote = quote.upper()
    if quote=="USD": quote="USDC"   # USDはUSDC扱い
    try:
        return float(os.getenv(f"DEX_NOTIONAL_{quote}", os.getenv("DEX_NOTIONAL", "1000")))
    except Exception:
        return 1000.0

class Dex0x(Adapter):
    name = "dex0x"
    BASE = os.getenv("ZEROX_API", "https://api.0x.org")
    KEY  = os.getenv("ZEROX_API_KEY", "")

    def _quote(self, sell_addr: str, buy_addr: str, sell_amount_int: int):
        url = f"{self.BASE}/swap/v1/quote"
        headers = {"0x-api-key": self.KEY} if self.KEY else {}
        params = {"sellToken": sell_addr, "buyToken": buy_addr, "sellAmount": str(sell_amount_int)}
        r = SESSION.get(url, params=params, headers=headers, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        try: return r.json()
        except Exception: return None

    @staticmethod
    def _route_str(qobj) -> str:
        try:
            parts = []
            for s in qobj.get("sources", []):
                prop = parse_f(s.get("proportion"))
                if prop > 0:
                    name = str(s.get("name","")).strip()
                    if name:
                        parts.append(f"{name}:{prop:.2f}")
            return "+".join(parts) if parts else ""
        except Exception:
            return ""

    def fetch(self):
        out={}
        pairs=[p.strip().upper() for p in os.getenv("DEX_PAIRS","ETH_USDT,WBTC_USDT,ETH_USDC,WBTC_USDC").split(",") if p.strip()]
        for p in pairs:
            if "_" not in p: continue
            base_sym, quote_sym = p.split("_",1)
            if quote_sym=="USD": quote_sym="USDC"
            if quote_sym not in QUOTES: continue
            b = TOKENS.get(base_sym); q = TOKENS.get(quote_sym)
            if not b or not q: continue

            notional = int(round(dex_notional_for_quote(quote_sym) * (10**q["decimals"])))
            q_buy  = self._quote(q["address"], b["address"], notional)     # クオート→ベース（買い）ASK
            if not q_buy: continue
            buy_base = int(q_buy.get("buyAmount") or 0)
            if buy_base <= 0: continue
            ask_price = notional / buy_base

            q_sell = self._quote(b["address"], q["address"], buy_base)     # ベース→クオート（売り）BID
            if not q_sell: continue
            buy_quote = int(q_sell.get("buyAmount") or 0)
            if buy_quote <= 0: continue
            bid_price = buy_quote / buy_base

            out[(base_sym if base_sym!="WETH" else "ETH", quote_sym)] = (bid_price, ask_price, None)

            if SHOW_DEX_ROUTE:
                r_buy  = self._route_str(q_buy)
                r_sell = self._route_str(q_sell)
                if r_buy and r_sell and r_buy != r_sell:
                    ROUTES[(base_sym if base_sym!="WETH" else "ETH", quote_sym)] = f"buy:{r_buy} | sell:{r_sell}"
                else:
                    ROUTES[(base_sym if base_sym!="WETH" else "ETH", quote_sym)] = r_buy or r_sell or "n/a"
        return out

# ====== ヘルパ ======
def min_qvol_for_quote(q: str) -> float:
    return float(MIN_QVOL.get(q.upper(), 0.0))

# ====== 集計 ======
# {(base,quote): {exchange: (bid,ask,qvol)}}
def aggregate(adapters: List[Adapter]):
    prices=defaultdict(dict)
    for ad in adapters:
        try:
            book=ad.fetch()
            for k,v in book.items():
                prices[k][ad.name]=v
        except Exception:
            pass
    return prices

# ====== スプレッド計算 ======
def compute_spreads(prices: Dict[Tuple[str,str], Dict[str, Tuple[float,float, Optional[float]]]]):
    rows=[]
    for (base,quote),mp in prices.items():
        if len(mp) < MIN_EXCH_FOR_PAIR:
            continue

        # 出来高フィルタ（板単位）
        qmin = min_qvol_for_quote(quote)
        vol_ok={}
        for ex,(bid,ask,qv) in mp.items():
            if bid<=0 or ask<=0:
                continue
            if qmin <= 0:
                vol_ok[ex]=(bid,ask,qv)
            else:
                if qv is None and not ALLOW_UNKNOWN_QVOL:
                    continue
                if (qv is None) or (qv >= qmin):
                    vol_ok[ex]=(bid,ask,qv)
        if len(vol_ok) < MIN_EXCH_FOR_PAIR:
            continue

        # ミッド中央値ベースで外れ値除去
        mids=[(b+a)/2 for (b,a,_) in vol_ok.values() if b>0 and a>0]
        if not mids: continue
        med=statistics.median(mids)
        tol=med*OUTLIER_TOL_PCT/100.0
        filtered={ex:(b,a,qv) for ex,(b,a,qv) in vol_ok.items() if abs(((b+a)/2)-med) <= tol}
        if len(filtered) < MIN_EXCH_FOR_PAIR:
            continue

        # 最良BID(売る先) / 最良ASK(買う先)
        best_buy_ex, best_buy = None, (0.0, 0.0, None)        # 高いBID
        best_sell_ex, best_sell = None, (0.0, float('inf'), None)  # 低いASK
        for ex,(bid,ask,qv) in filtered.items():
            if bid > best_buy[0]:
                best_buy, best_buy_ex = (bid,ask,qv), ex
            if ask < best_sell[1]:
                best_sell, best_sell_ex = (bid,ask,qv), ex

        if not best_buy_ex or not best_sell_ex: continue
        if best_buy_ex == best_sell_ex:       # 同一所内は除外（クロス所アーブのみ）
            continue
        if best_buy[0] <= 0 or best_sell[1] <= 0: continue

        spread_pct = (best_buy[0]/best_sell[1] - 1.0)*100.0
        if spread_pct <= MIN_DETECT_SPREAD_PCT:    # ←★ 検出フィルタ：0.8%以下は捨てる
            continue
        if spread_pct > MAX_SPREAD_CAP_PCT:
            continue
        # 価格比の健全性（売買双方のASKの比でガード）
        if best_buy[1] > 0 and (best_sell[1] / best_buy[1]) > REQUIRE_RATIO_MAX:
            continue

        # 出来高しきい値を両サイド要求するか
        qmin = min_qvol_for_quote(quote)
        if qmin > 0 and REQUIRE_QVOL_BOTH_SIDES:
            if not ((best_sell[2] is not None and best_sell[2] >= qmin) and (best_buy[2] is not None and best_buy[2] >= qmin)):
                continue

        rows.append((base,quote,spread_pct,(best_sell_ex,best_sell[1]),(best_buy_ex,best_buy[0])))
    rows.sort(key=lambda x:x[2], reverse=True)
    return rows

# ====== 実行 ======
def build_adapters():
    name2cls = {
        "bybit": Bybit, "mexc": MEXC, "bingx": BingX, "bitget": Bitget,
        "bitflyer": Bitflyer, "bitbank": Bitbank, "coincheck": Coincheck, "gmocoin": GMOCoin,
        "dex0x": Dex0x
    }
    return [name2cls[n]() for n in ENABLED_EXCHANGES if n in name2cls]

if __name__ == "__main__":
    print(f"[CONFIG] EXCH={ENABLED_EXCHANGES} QUOTES={sorted(QUOTES)} INTERVAL_S={INTERVAL_S}")
    print(f"[FILTER] MIN_EXCH_FOR_PAIR={MIN_EXCH_FOR_PAIR} OUTLIER_TOL_PCT={OUTLIER_TOL_PCT}% SPREAD_CAP={MAX_SPREAD_CAP_PCT}% RATIO_MAX={REQUIRE_RATIO_MAX}x MIN_BASE_LEN={MIN_BASE_LEN}")
    print(f"[ALERT] base={ALERT_BASE_PCT:.2f}% + fee={FEE_BUFFER_PCT:.2f}% => alert >= {ALERT_EFF_PCT:.2f}%")
    thr = {k: v for k, v in MIN_QVOL.items() if v > 0}
    print(f"[VOLUME] thresholds={thr} require_both={REQUIRE_QVOL_BOTH_SIDES} allow_unknown={ALLOW_UNKNOWN_QVOL} show_dex_route={SHOW_DEX_ROUTE}")
    print(f"[DETECT] min_spread={MIN_DETECT_SPREAD_PCT:.2f}%  (rows below this are filtered out)")

    adapters = build_adapters()
    if not adapters:
        print("No exchanges enabled. Set ENABLED_EXCHANGES.")
        sys.exit(1)

    while True:
        prices = aggregate(adapters)
        rows = compute_spreads(prices)

        print(f"          Top spreads (ASK→BID, detect >= {MIN_DETECT_SPREAD_PCT:.2f}% | alert >= {ALERT_EFF_PCT:.2f}%)")
        table=[]
        for base,quote,sp,buy,sell in rows[:TOPN]:
            table.append([base,quote,f"{sp:7.3f}", f"{buy[0]} / {buy[1]:.8f}", f"{sell[0]} / {sell[1]:.8f}",
                          ROUTES.get((base,quote),"") if SHOW_DEX_ROUTE else ""])
        headers = ["Coin","Quote","Spread%","Buy@ (ex / price)","Sell@ (ex / price)"]
        if SHOW_DEX_ROUTE: headers.append("DEX route (0x)")
        if table:
            print(tabulate(table, headers=headers, tablefmt="heavy_outline"))
        else:
            print("(no opportunities after filters)")

        for base,quote,sp,buy,sell in rows:
            if sp >= ALERT_EFF_PCT:
                print(f"[ALERT] {base}/{quote} {sp:.2f}%  BUY {buy[0]} @{buy[1]:.8f} -> SELL {sell[0]} @{sell[1]:.8f}")
                if SHOW_DEX_ROUTE and (base,quote) in ROUTES:
                    print(f"        [dex0x route] {ROUTES[(base,quote)]}")
                beep()

        time.sleep(INTERVAL_S)
