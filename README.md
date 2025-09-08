# BB Bot（Bybit / pybit / Render Worker）
- ボリンジャー **端→ミドル→次バーエントリー** 戦略ボット
- **1日最大10回**の新規エントリー制限（UTC）

## 同梱
- `bb_bot_pybit.py` … ボット本体
- `requirements.txt`
- `render.yaml` … Render の Worker 定義（永続ディスクで `bot_state.json` を保存）

## 主要環境変数
- `BYBIT_API_KEY` / `BYBIT_API_SECRET`（必須）
- `BB_TESTNET=1` テスト / `0` 本番
- 戦略: `BB_INTERVAL=60`, `BB_N=12`, `BB_K=2.0`, `BB_TIME_STOP_BARS=12`
- コスト: `BB_FEE_BPS=7.5`, `BB_SLIP_BPS=5`
- リスク: `BB_LEVERAGE=5`, `BB_RISK_FRAC=0.005`
- 安全装置: `BB_MAX_TRADES_PER_DAY=10`

## 使い方（ローカル）
```powershell
$env:BYBIT_API_KEY="xxxx"
$env:BYBIT_API_SECRET="yyyy"
$env:BB_TESTNET="0"
$env:BB_MAX_TRADES_PER_DAY="10"
python .\bb_bot_pybit.py
```

## Render (Blueprint)
- リポにこの3ファイルを置いて、Render で **New → Blueprint**。
- 環境変数に API キーを追加して Deploy。
