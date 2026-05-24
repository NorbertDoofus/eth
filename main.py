"""
Telegram-бот + торговый pipeline в одном файле.
Запуск: python bot.py
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

# ============================================================
#  НАСТРОЙКИ
# ============================================================
BOT_TOKEN  = "6320815494:AAG9H74_xhhsNOen95YD-AtKuCfaHSXjplQ"   # получить у @BotFather
SYMBOL     = "ETHUSDT"
START_DATE = "2026-05-01"
END_DATE   = "2026-05-25"
RR         = 2                        # риск/прибыль

# ============================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
session = HTTP(testnet=False)


# ============================================================
#  ШАГ 1 — ЗАГРУЗКА СВЕЧЕЙ С BYBIT
# ============================================================

def _get_chunk(symbol, interval, start_ts, end_ts, retries=3):
    for attempt in range(1, retries + 1):
        try:
            resp = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=start_ts,
                end=end_ts,
                limit=1000,
            )
            return resp.get("result", {}).get("list", [])
        except Exception as exc:
            logger.warning("Попытка %d/%d: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2)


def _fetch_day(symbol, interval, day_start):
    """Загружает 1 день, разбивая на 4 части по 6 часов."""
    seen, candles = set(), []
    for i in range(4):
        ps = day_start + timedelta(hours=i * 6)
        pe = day_start + timedelta(hours=(i + 1) * 6) - timedelta(milliseconds=1)
        start_ts = int(ps.timestamp() * 1000)
        end_ts   = int(pe.timestamp() * 1000)
        last_ts  = -1

        while True:
            chunk = _get_chunk(symbol, interval, start_ts, end_ts)
            if not chunk:
                break
            chunk = sorted(chunk, key=lambda x: int(x[0]))
            for c in chunk:
                if c[0] not in seen:
                    seen.add(c[0])
                    candles.append(c)
            new_ts = int(chunk[-1][0])
            if new_ts == last_ts or len(chunk) < 1000:
                break
            last_ts  = new_ts
            start_ts = new_ts + 1
            time.sleep(0.1)
        time.sleep(0.3)
    return candles


def fetch_symbol(symbol, interval):
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date   = datetime.strptime(END_DATE,   "%Y-%m-%d")
    all_data   = []
    cur = start_date

    logger.info("Загрузка %s TF=%sm ...", symbol, interval)
    while cur <= end_date:
        day_candles = _fetch_day(symbol, interval, cur)
        all_data.extend(day_candles)
        logger.info("  %s: %d свечей", cur.date(), len(day_candles))
        cur += timedelta(days=1)
        time.sleep(0.5)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info("Загружено %d свечей TF=%sm", len(df), interval)
    return df


# ============================================================
#  ШАГ 2 — ПИН-БАРЫ
# ============================================================

def add_pinbars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = abs(df["close"] - df["open"])
    df["upshadow"] = np.where(
        df["open"] > df["close"],
        df["high"] - df["open"],
        df["high"] - df["close"],
    )
    df["downshadow"] = np.where(
        df["open"] > df["close"],
        df["close"] - df["low"],
        df["open"]  - df["low"],
    )
    df["body_prev"]  = df["body"].shift(1)
    df["open_prev"]  = df["open"].shift(1)
    df["close_prev"] = df["close"].shift(1)

    df["bearish_pinbar"] = np.where(
        (df["close_prev"] > df["open_prev"]) &
        (df["body_prev"]  > df["body"]) &
        (df["upshadow"]   > 0.5 * df["body"]) &
        (df["upshadow"]   > 2   * df["downshadow"]),
        1, 0,
    )
    df["bullish_pinbar"] = np.where(
        (df["open_prev"]  > df["close_prev"]) &
        (df["body_prev"]  > df["body"]) &
        (df["downshadow"] > 0.5 * df["body"]) &
        (df["downshadow"] > 2   * df["upshadow"]),
        1, 0,
    )
    return df


def process_pinbars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_pinbars(df)
    logger.info("Пин-бары: bearish=%d bullish=%d",
                int(df["bearish_pinbar"].sum()), int(df["bullish_pinbar"].sum()))
    return df


# ============================================================
#  ШАГ 3 — БЭКТЕСТ
# ============================================================

def _check_tp(df5, start_index, tp):
    for k in range(start_index, len(df5)):
        if df5.iloc[k]["high"] >= tp:
            return True, df5.iloc[k]["timestamp"]
    return False, None


def run_backtest(df5: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    TIME_COL = "timestamp"
    for df in (df5, df1):
        for col in ["open", "high", "low", "close", "volume", "bullish_pinbar"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df5 = df5.sort_values(TIME_COL).reset_index(drop=True)
    df1 = df1.sort_values(TIME_COL).set_index(TIME_COL)

    trades = []
    i = 1
    while i < len(df5) - 1:
        prev, mid, nxt = df5.iloc[i - 1], df5.iloc[i], df5.iloc[i + 1]

        if prev["high"] < nxt["low"]:
            ob_time = mid[TIME_COL]
            ob_low  = prev["high"]
            ob_high = nxt["low"]

            reaction_index = None
            violated = False

            for j in range(i + 2, len(df5)):
                c = df5.iloc[j]
                if c["open"] > ob_high and c["close"] > ob_high and c["low"] < ob_low:
                    reaction_index = j
                    break
                if c["low"] < ob_high:
                    violated = True
                    break

            if reaction_index is not None and not violated:
                r5m = df5.iloc[reaction_index]
                rt  = r5m[TIME_COL]
                df1_slice = df1.loc[rt: rt + timedelta(minutes=4)]

                confirmed = any(
                    row["open"]  > ob_high and
                    row["close"] > ob_high and
                    row["low"]   < ob_low  and
                    int(row["bullish_pinbar"]) == 1
                    for _, row in df1_slice.iterrows()
                )

                if confirmed:
                    entry = r5m["close"]
                    risk  = entry - prev["low"]
                    tp    = entry + risk * RR
                    tp_hit, tp_time = _check_tp(df5, reaction_index + 1, tp)
                    trades.append({
                        "ob_time":    ob_time,
                        "entry_time": rt,
                        "entry":      entry,
                        "tp":         tp,
                        "tp_hit":     tp_hit,
                        "tp_time":    tp_time,
                    })
        i += 1

    return pd.DataFrame(trades)


# ============================================================
#  ГЛАВНЫЙ PIPELINE
# ============================================================

def run_full_pipeline() -> dict:
    logger.info("=== PIPELINE START ===")

    # 1. Загрузка свечей — только в память
    df1_raw = fetch_symbol(SYMBOL, "1")
    df5_raw = fetch_symbol(SYMBOL, "5")

    # 2. Пин-бары — только в память
    df1_pb = process_pinbars(df1_raw)
    df5_pb = process_pinbars(df5_raw)

    # 3. Бэктест — только в память
    trades = run_backtest(df5_pb, df1_pb)

    total  = len(trades)
    wins   = int((trades["tp_hit"] == True).sum()) if total else 0
    losses = total - wins
    wr     = wins / total * 100 if total else 0.0

    logger.info("=== PIPELINE DONE trades=%d wins=%d wr=%.1f%% ===", total, wins, wr)
    return {
        "trades":  trades,
        "total":   total,
        "wins":    wins,
        "losses":  losses,
        "winrate": wr,
        "period":  f"{START_DATE} → {END_DATE}",
    }


# ============================================================
#  TELEGRAM БОТ
# ============================================================

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[InlineKeyboardButton("📊 Анализ", callback_data="run_analysis")]]
    await update.message.reply_text(
        "👋 *Торговый бот*\n\n"
        f"Символ: `{SYMBOL}`\n"
        f"Период: `{START_DATE}` → `{END_DATE}`\n"
        f"RR: `1:{RR}`\n\n"
        "Нажмите кнопку, чтобы запустить анализ:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cb_analysis(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    status = await query.message.reply_text(
        "⏳ *Запускаю анализ...*\n\n"
        "1️⃣ Загрузка свечей с Bybit\n"
        "2️⃣ Расчёт пин-баров\n"
        "3️⃣ Бэктест\n\n"
        "_Это займёт несколько минут_",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_full_pipeline)
        await status.delete()
        await _send_report(query.message, result)
    except Exception as exc:
        logger.exception("Pipeline error")
        await status.edit_text(
            f"❌ Ошибка:\n\n`{exc}`",
            parse_mode=ParseMode.MARKDOWN,
        )


async def _send_report(message, result: dict) -> None:
    trades = result["trades"]
    total  = result["total"]
    wins   = result["wins"]
    losses = result["losses"]
    wr     = result["winrate"]

    # ── Сообщение 1: Статистика ──────────────────────────────
    if wr >= 60:
        verdict = "🟢 Стратегия показывает хорошие результаты."
    elif wr >= 45:
        verdict = "🟡 Результаты умеренные, стоит доработать фильтры."
    elif total == 0:
        verdict = "⚠️ Сделок не найдено. Попробуйте изменить параметры."
    else:
        verdict = "🔴 Winrate низкий. Рекомендуется пересмотр стратегии."

    await message.reply_text(
        "📊 *ОТЧЁТ ПО АНАЛИЗУ*\n"
        f"{'─' * 30}\n\n"
        f"🗓 Период: `{result['period']}`\n"
        f"💱 Символ: `{SYMBOL}`  |  RR: `1:{RR}`\n\n"
        f"📈 *Всего сделок:*     `{total}`\n"
        f"✅ *TP достигнут:*    `{wins}`\n"
        f"❌ *TP не достигнут:* `{losses}`\n\n"
        f"🎯 *Winrate:* `{wr:.1f}%`\n\n"
        f"{verdict}",
        parse_mode=ParseMode.MARKDOWN,
    )

    # ── Сообщение 2: Открытые сделки ────────────────────────
    if trades.empty:
        return

    open_trades = trades[trades["tp_hit"] == False]
    if open_trades.empty:
        await message.reply_text(
            "✅ *Все сделки закрыты* — открытых позиций нет.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = [f"📋 *СДЕЛКИ БЕЗ ТЕЙК-ПРОФИТА* ({len(open_trades)} шт.)\n" + "─" * 30]

    for i, (_, row) in enumerate(open_trades.iterrows(), 1):
        entry = row.get("entry", 0)
        tp    = row.get("tp", 0)
        lines.append(
            f"\n*#{i}*\n"
            f"  🕐 OB:    `{str(row.get('ob_time', '—'))[:16]}`\n"
            f"  📥 Вход:  `{str(row.get('entry_time', '—'))[:16]}`\n"
            f"  💰 Цена:  `{entry:.4f}`\n"
            f"  🎯 TP:    `{tp:.4f}`\n"
            f"  📏 До TP: `+{tp - entry:.4f}`"
        )

        # Отправляем пачками по 10, чтобы не превысить лимит Telegram
        if i % 10 == 0:
            await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
            lines = ["📋 *ПРОДОЛЖЕНИЕ...*\n" + "─" * 30]

    if len(lines) > 1:
        await message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ============================================================
#  ТОЧКА ВХОДА
# ============================================================

def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(cb_analysis, pattern="^run_analysis$"))
    logger.info("Бот запущен.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
