import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


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
    required = ["Date", "Time", "Currency", "Event_Normalized", "Impact", "Relevance", "Previous", "Consensus", "Actual"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing event columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).copy()
    df = df.drop_duplicates(
        subset=["datetime", "Currency", "Event_Normalized", "Impact", "Previous", "Consensus", "Actual"]
    )
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(show_spinner="Loading XAUUSD M1 data...")
def load_gold():
    df = pd.read_csv(GOLD_FILE, sep="\t", low_memory=False)
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
    required = ["date", "time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing gold columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        errors="coerce",
    )
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    # Force nanosecond precision explicitly. Pandas 2.x may otherwise keep datetime64[us],
    # and converting that to raw integers would make the timestamps look like 1970.
    df["datetime"] = df["datetime"].astype("datetime64[ns]")
    times = df["datetime"].to_numpy(dtype="datetime64[ns]")
    prices = df["close"].to_numpy(dtype="float64")
    return df[["datetime", "open", "high", "low", "close"]], times, prices


def price_at_or_before(times, prices, target_ns):
    idx = np.searchsorted(times, target_ns, side="right") - 1
    return np.nan if idx < 0 else float(prices[idx])


def price_at_or_after(times, prices, target_ns):
    idx = np.searchsorted(times, target_ns, side="left")
    return np.nan if idx >= len(prices) else float(prices[idx])


def event_chart(event_time, gold_df, reaction_minutes):
    start = event_time - pd.Timedelta(minutes=30)
    end = event_time + pd.Timedelta(minutes=max(60, reaction_minutes))
    part = gold_df[(gold_df["datetime"] >= start) & (gold_df["datetime"] <= end)].copy()
    if part.empty:
        return None
    fig = go.Figure(data=[go.Candlestick(x=part["datetime"], open=part["open"], high=part["high"], low=part["low"], close=part["close"], name="XAUUSD M1")])
    fig.add_vline(x=event_time, line_dash="dash", annotation_text="EVENT")
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def analyze(filtered_events, gold_times, gold_prices, reaction_minutes):
    if filtered_events.empty:
        return pd.DataFrame()

    events_work = filtered_events.copy()
    events_work["datetime"] = pd.to_datetime(events_work["datetime"], errors="coerce").astype("datetime64[ns]")
    events_work = events_work.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    gold_lookup = pd.DataFrame({
        "gold_datetime": pd.to_datetime(gold_times, errors="coerce").astype("datetime64[ns]"),
        "gold_close": gold_prices,
    }).sort_values("gold_datetime").reset_index(drop=True)

    before = pd.merge_asof(
        events_work[["datetime"]],
        gold_lookup,
        left_on="datetime",
        right_on="gold_datetime",
        direction="backward",
    )

    targets = events_work[["datetime"]].copy()
    targets["target_datetime"] = targets["datetime"] + pd.Timedelta(minutes=reaction_minutes)

    after = pd.merge_asof(
        targets[["target_datetime"]],
        gold_lookup,
        left_on="target_datetime",
        right_on="gold_datetime",
        direction="forward",
    )

    result = events_work.copy()
    result["price_before"] = before["gold_close"].to_numpy()
    result["price_after"] = after["gold_close"].to_numpy()
    result = result.dropna(subset=["price_before", "price_after"]).copy()

    result["movement"] = result["price_after"] - result["price_before"]
    result["points"] = result["movement"] * 100
    result["direction"] = np.where(
        result["movement"] > 0, "UP",
        np.where(result["movement"] < 0, "DOWN", "FLAT")
    )
    result["date"] = result["datetime"].dt.strftime("%Y-%m-%d")
    result["time"] = result["datetime"].dt.strftime("%H:%M")
    result = result.rename(columns={"Event_Normalized": "event", "Currency": "currency", "Impact": "impact", "Relevance": "relevance",
                                    "Previous": "previous", "Consensus": "consensus", "Actual": "actual"})
    result.attrs["before_ok"] = int(before["gold_close"].notna().sum())
    result.attrs["after_ok"] = int(after["gold_close"].notna().sum())
    return result


def main():
    st.title("📈 XAUUSD Economic Events Analysis")
    st.caption("واجهة مباشرة لملفات CSV — بدون Swagger وبدون API")

    if not os.path.exists(EVENTS_FILE) or not os.path.exists(GOLD_FILE):
        st.error("ملفات البيانات غير موجودة في backend/data/")
        st.code("backend/data/XAUUSD_events.csv\nbackend/data/XAUUSDm_M1.csv")
        return

    try:
        events = load_events()
        gold_df, gold_times, gold_prices = load_gold()
    except Exception as exc:
        st.error(f"تعذر تحميل البيانات: {exc}")
        return

    min_date = events["datetime"].min().date()
    max_date = events["datetime"].max().date()

    with st.sidebar:
        st.header("🎛️ Filters")
        currencies = sorted(events["Currency"].dropna().astype(str).unique().tolist())
        impacts = sorted(events["Impact"].dropna().astype(str).unique().tolist())
        relevances = sorted(events["Relevance"].dropna().astype(str).unique().tolist())
        event_names = sorted(events["Event_Normalized"].dropna().astype(str).unique().tolist())

        currency = st.selectbox("Currency", ["All"] + currencies)
        impact = st.selectbox("Impact", ["All"] + impacts)
        relevance = st.selectbox("Relevance", ["All"] + relevances)
        selected_events = st.multiselect("Event", event_names, default=[], placeholder="All events")
        if selected_events and st.button("✕ Clear Event Selection", use_container_width=True):
            st.session_state["clear_events"] = True
            st.rerun()
        reaction = st.selectbox("Reaction period", [1, 5, 15, 30, 60], index=2, format_func=lambda x: f"{x} minutes")
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        chart_count = st.slider("Mini charts", 3, 24, 12)

        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
        reset_clicked = st.button("↻ Reset", use_container_width=True)

    if st.session_state.pop("clear_events", False):
        st.info("تم إلغاء اختيار الحدث. اضغط Analyze لعرض جميع الأحداث المطابقة لباقي الفلاتر.")

    if reset_clicked:
        for key in ["analysis", "reaction", "filtered_count", "filter_info", "filter_steps"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.info(
        f"Gold M1: {len(gold_df):,} rows  •  Events: {len(events):,} rows  •  Data: {min_date} → {max_date}"
    )

    if analyze_clicked:
        filtered = events.copy()
        filter_info = []
        filter_steps = [("All events", len(filtered))]

        if currency != "All":
            filtered = filtered[filtered["Currency"] == currency]
            filter_info.append(f"Currency={currency}")
            filter_steps.append((f"Currency = {currency}", len(filtered)))
        if impact != "All":
            filtered = filtered[filtered["Impact"] == impact]
            filter_info.append(f"Impact={impact}")
            filter_steps.append((f"Impact = {impact}", len(filtered)))
        if relevance != "All":
            filtered = filtered[filtered["Relevance"] == relevance]
            filter_info.append(f"Relevance={relevance}")
            filter_steps.append((f"Relevance = {relevance}", len(filtered)))
        if selected_events:
            filtered = filtered[filtered["Event_Normalized"].isin(selected_events)]
            filter_info.append(f"Events={len(selected_events)}")
            filter_steps.append((f"Selected events ({len(selected_events)})", len(filtered)))

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered["datetime"].dt.date >= start_date)
                & (filtered["datetime"].dt.date <= end_date)
            ]
            filter_info.append(f"Date={start_date} → {end_date}")
            filter_steps.append((f"Date = {start_date} → {end_date}", len(filtered)))

        st.session_state["filtered_count"] = len(filtered)
        st.session_state["filter_info"] = " • ".join(filter_info) if filter_info else "No filters — all events"
        st.session_state["filter_steps"] = filter_steps

        with st.spinner("Analyzing XAUUSD reactions..."):
            result = analyze(filtered, gold_times, gold_prices, reaction)

        st.session_state["analysis"] = result
        st.session_state["reaction"] = reaction

    # Always show a real XAUUSD M1 candlestick preview when events are available.
    # This preview is independent of the reaction calculation, so a matching-reaction
    # problem cannot hide the underlying market chart.
    if "filtered_count" in st.session_state and st.session_state.get("filtered_count", 0) > 0:
        st.subheader("🕯️ XAUUSD M1 Chart Preview")
        preview_events = events.copy()
        # Pick an event that actually has nearby M1 candles instead of the first
        # calendar event, which may fall on a weekend/market-closed period.
        event_ns = preview_events["datetime"].astype("datetime64[ns]").astype("int64").to_numpy()
        gold_ns = gold_df["datetime"].astype("datetime64[ns]").astype("int64").to_numpy()
        distances = np.abs(event_ns[:, None] - gold_ns[::max(1, len(gold_ns) // 20000)][None, :])
        preview_event = preview_events.iloc[int(np.argmin(distances).item() // distances.shape[1])]
        preview_chart = event_chart(preview_event["datetime"], gold_df, reaction)
        if preview_chart is not None:
            st.caption(
                f"Event: {preview_event['Event_Normalized']} • "
                f"{preview_event['datetime'].strftime('%Y-%m-%d %H:%M:%S')} • "
                f"Window: 30 min before → {max(60, reaction)} min after"
            )
            st.plotly_chart(
                preview_chart,
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": True},
            )
        else:
            st.warning("لم يتم العثور على شموع M1 داخل نافذة الشارت لهذا الحدث.")

    result = st.session_state.get("analysis", pd.DataFrame())
    reaction = st.session_state.get("reaction", reaction)

    if result.empty:
        if "filtered_count" in st.session_state:
            filtered_count = st.session_state["filtered_count"]
            filter_info = st.session_state.get("filter_info", "")
            if filtered_count == 0:
                st.warning("لا توجد أحداث بعد تطبيق الفلاتر.")
                st.caption(f"الفلاتر الحالية: {filter_info}")
                st.info("جرّب Clear Event Selection ثم Analyze. جدول التشخيص يوضح عند أي فلتر أصبح العدد صفر.")
                if "filter_steps" in st.session_state:
                    st.markdown("#### 🔎 تشخيص الفلاتر")
                    st.dataframe(pd.DataFrame(st.session_state["filter_steps"], columns=["Filter step", "Events remaining"]), use_container_width=True, hide_index=True)
            else:
                st.warning(f"تم العثور على {filtered_count:,} حدث، لكن لم يتم العثور على شمعة ذهب مطابقة لفترة التفاعل.")
                st.caption(f"الفلاتر الحالية: {filter_info}")
                st.markdown("#### 🔎 تشخيص بيانات الوقت")
                st.write(f"Gold M1 range: **{gold_df["datetime"].min()} → {gold_df["datetime"].max()}**")
                st.write(f"Events range: **{events["datetime"].min()} → {events["datetime"].max()}**")
                st.write(f"Reaction: **{reaction} minutes**")
                sample_times = events["datetime"].dropna().head(5).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                st.write(f"Sample event times: **{sample_times}**")
        else:
            st.markdown("### اختر الفلاتر من الجانب ثم اضغط **Analyze** لعرض النتائج.")
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
        for col, (_, item) in zip(cols, chart_rows.iloc[start:start + 3].iterrows()):
            with col:
                arrow = "↑" if item["movement"] > 0 else "↓" if item["movement"] < 0 else "→"
                st.markdown(
                    f"**{item['event']}**\n"
                    f"{item['date']} {item['time']} • {item['impact']}\n"
                    f"**{item['movement']:+.2f} USD {arrow}**"
                )
                chart = event_chart(item["datetime"], gold_df, reaction)
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    st.subheader("📋 Detailed Results")
    display = result[[
        "date", "time", "currency", "event", "impact", "previous", "consensus", "actual",
        "price_before", "price_after", "movement", "points", "direction",
    ]].copy()
    display.columns = [
        "Date", "Time", "Currency", "Event", "Impact", "Previous", "Consensus", "Actual",
        "Price Before", "Price After", "Movement USD", "Points", "Direction",
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", data=csv, file_name="XAUUSD_event_analysis.csv", mime="text/csv")


if __name__ == "__main__":
    main()
