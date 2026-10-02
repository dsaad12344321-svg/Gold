import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="XAUUSD Economic Analysis",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_FILE = os.path.join(BASE_DIR, "backend", "data", "XAUUSD_events.csv")
GOLD_FILE = os.path.join(BASE_DIR, "backend", "data", "XAUUSDm_M1.csv")


@st.cache_data(show_spinner="Loading economic events...")
def load_events():
    df = pd.read_csv(EVENTS_FILE, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Date",
        "Time",
        "Currency",
        "Event_Normalized",
        "Impact",
        "Relevance",
        "Previous",
        "Consensus",
        "Actual",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing event columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )

    df = df.dropna(subset=["datetime"]).copy()
    df = df.drop_duplicates(
        subset=[
            "datetime",
            "Currency",
            "Event_Normalized",
            "Impact",
            "Previous",
            "Consensus",
            "Actual",
        ]
    )
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(show_spinner="Loading XAUUSD M1 data...")
def load_gold():
    df = pd.read_csv(
        GOLD_FILE,
        sep="\t",
        low_memory=False,
    )
    df.columns = [
        str(c).strip().strip("<>").lower()
        for c in df.columns
    ]

    required = ["date", "time", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing gold columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip()
        + " "
        + df["time"].astype(str).str.strip(),
        errors="coerce",
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Numpy arrays make event-to-price lookups very fast.
    times = df["datetime"].astype("int64").to_numpy()
    prices = df["close"].to_numpy(dtype="float64")
    return df[["datetime", "close"]], times, prices


def price_at_or_before(times, prices, target_ns):
    idx = np.searchsorted(times, target_ns, side="right") - 1
    if idx < 0:
        return np.nan
    return float(prices[idx])


def price_at_or_after(times, prices, target_ns):
    idx = np.searchsorted(times, target_ns, side="left")
    if idx >= len(prices):
        return np.nan
    return float(prices[idx])


def sparkline(event_time, gold_df, reaction_minutes):
    start = event_time - pd.Timedelta(minutes=30)
    end = event_time + pd.Timedelta(minutes=max(60, reaction_minutes))
    part = gold_df[
        (gold_df["datetime"] >= start)
        & (gold_df["datetime"] <= end)
    ][["datetime", "close"]].copy()

    if part.empty:
        return None

    # Keep the mini chart light.
    if len(part) > 90:
        idx = np.linspace(0, len(part) - 1, 90).astype(int)
        part = part.iloc[np.unique(idx)]

    part = part.set_index("datetime")
    return part


def analyze(filtered_events, gold_df, gold_times, gold_prices, reaction_minutes):
    rows = []

    for row in filtered_events.itertuples(index=False):
        event_time = row.datetime
        before = price_at_or_before(
            gold_times,
            gold_prices,
            event_time.value,
        )
        after_time = event_time + pd.Timedelta(minutes=reaction_minutes)
        after = price_at_or_after(
            gold_times,
            gold_prices,
            after_time.value,
        )

        if np.isnan(before) or np.isnan(after):
            continue

        movement = after - before

        rows.append(
            {
                "datetime": event_time,
                "date": event_time.strftime("%Y-%m-%d"),
                "time": event_time.strftime("%H:%M"),
                "currency": row.Currency,
                "event": row.Event_Normalized,
                "impact": row.Impact,
                "relevance": row.Relevance,
                "previous": row.Previous,
                "consensus": row.Consensus,
                "actual": row.Actual,
                "price_before": before,
                "price_after": after,
                "movement": movement,
                "points": movement * 100,
                "direction": (
                    "UP" if movement > 0
                    else "DOWN" if movement < 0
                    else "FLAT"
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    st.title("📈 XAUUSD Economic Events Analysis")
    st.caption("واجهة مباشرة لملفات CSV — بدون Swagger وبدون API")

    if not os.path.exists(EVENTS_FILE) or not os.path.exists(GOLD_FILE):
        st.error("ملفات البيانات غير موجودة في backend/data/")
        st.code(
            "backend/data/XAUUSD_events.csv\n"
            "backend/data/XAUUSDm_M1.csv"
        )
        return

    try:
        events = load_events()
        gold_df, gold_times, gold_prices = load_gold()
    except Exception as exc:
        st.error(f"تعذر تحميل البيانات: {exc}")
        return

    with st.sidebar:
        st.header("🎛️ Filters")

        currencies = sorted(events["Currency"].dropna().unique().tolist())
        impacts = sorted(events["Impact"].dropna().unique().tolist())
        relevances = sorted(events["Relevance"].dropna().unique().tolist())
        event_names = sorted(
            events["Event_Normalized"].dropna().unique().tolist()
        )

        currency = st.selectbox("Currency", ["All"] + currencies)
        impact = st.selectbox("Impact", ["All"] + impacts)
        relevance = st.selectbox("Relevance", ["All"] + relevances)

        selected_events = st.multiselect(
            "Event",
            event_names,
            placeholder="All events",
        )

        reaction = st.selectbox(
            "Reaction period",
            [1, 5, 15, 30, 60],
            index=2,
            format_func=lambda x: f"{x} minutes",
        )

        min_date = events["datetime"].min().date()
        max_date = events["datetime"].max().date()

        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        chart_count = st.slider(
            "Mini charts",
            min_value=3,
            max_value=24,
            value=12,
        )

        analyze_clicked = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True,
        )

        if st.button("↻ Reset", use_container_width=True):
            st.rerun()

    st.info(
        f"Gold M1: {len(gold_df):,} rows  •  "
        f"Events: {len(events):,} rows  •  "
        f"Data: {min_date} → {max_date}"
    )

    if not analyze_clicked and "analysis" not in st.session_state:
        st.markdown(
            "### اختر الفلاتر من الجانب ثم اضغط **Analyze** لعرض النتائج."
        )
        return

    if analyze_clicked:
        filtered = events.copy()

        if currency != "All":
            filtered = filtered[filtered["Currency"] == currency]
        if impact != "All":
            filtered = filtered[filtered["Impact"] == impact]
        if relevance != "All":
            filtered = filtered[filtered["Relevance"] == relevance]
        if selected_events:
            filtered = filtered[
                filtered["Event_Normalized"].isin(selected_events)
            ]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered["datetime"].dt.date >= start_date)
                & (filtered["datetime"].dt.date <= end_date)
            ]

        with st.spinner("Analyzing XAUUSD reactions..."):
            result = analyze(
                filtered,
                gold_df,
                gold_times,
                gold_prices,
                reaction,
            )

        st.session_state["analysis"] = result
        st.session_state["reaction"] = reaction

    result = st.session_state.get("analysis", pd.DataFrame())
    reaction = st.session_state.get("reaction", reaction)

    if result.empty:
        st.warning("لا توجد نتائج مطابقة للفلاتر المختارة.")
        return

    total = len(result)
    positive = int((result["movement"] > 0).sum())
    negative = int((result["movement"] < 0).sum())
    average = float(result["movement"].mean())
    positive_rate = positive / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{total:,}")
    c2.metric("Positive", f"{positive:,}")
    c3.metric("Negative", f"{negative:,}")
    c4.metric("Avg Movement", f"{average:+.2f} USD")

    st.caption(f"Positive rate: {positive_rate:.1f}%")

    st.subheader("📊 Event Charts")

    chart_rows = result.head(chart_count)

    for start in range(0, len(chart_rows), 3):
        cols = st.columns(3)

        for col, (_, item) in zip(
            cols,
            chart_rows.iloc[start:start + 3].iterrows(),
        ):
            with col:
                arrow = "↑" if item["movement"] > 0 else "↓" if item["movement"] < 0 else "→"
                st.markdown(
                    f"**{item['event']}**  
"
                    f"{item['date']} {item['time']} • {item['impact']}  
"
                    f"**{item['movement']:+.2f} USD {arrow}**"
                )
                chart = sparkline(
                    item["datetime"],
                    gold_df,
                    reaction,
                )
                if chart is not None:
                    st.line_chart(
                        chart,
                        y="close",
                        height=150,
                        use_container_width=True,
                    )

    st.subheader("📋 Detailed Results")

    display = result[
        [
            "date",
            "time",
            "currency",
            "event",
            "impact",
            "previous",
            "consensus",
            "actual",
            "price_before",
            "price_after",
            "movement",
            "points",
            "direction",
        ]
    ].copy()

    display.columns = [
        "Date",
        "Time",
        "Currency",
        "Event",
        "Impact",
        "Previous",
        "Consensus",
        "Actual",
        "Price Before",
        "Price After",
        "Movement USD",
        "Points",
        "Direction",
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results CSV",
        data=csv,
        file_name="XAUUSD_event_analysis.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
