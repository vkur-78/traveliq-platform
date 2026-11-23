"""
TRAVELIQ — Travel Intelligence Platform
Redesigned Streamlit Web Application (Phase 2).

Navigation Sections:
  1. Home Dashboard
  2. Flight Price Predictor       (connects to Flight REST API via HTTP)
  3. Trip & Hotel Recommendations (uses app.recommender hybrid package engine)
  4. Gender Classification Demo   (connects to Gender REST API via HTTP)
  5. Analytics                    (interactive travel & pricing insights)
"""

import base64
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.api_client import (
    check_api_health,
    request_flight_prediction,
    request_gender_prediction,
)
from app.config import DATASET_DIR, FLIGHT_API_URL, GENDER_API_URL, PROJECT_ROOT
from app.gender_features import EXPERIMENTAL_GENDER_NOTICE
from app.recommender import ONE_HOTEL_LIMITATION_NOTE, TravelPackageRecommender

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="TRAVELIQ | Travel Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = [
    "Home Dashboard",
    "Flight Price Predictor",
    "Trip & Hotel Recommendations",
    "Gender Classification Demo",
    "Analytics",
]

NAV_ICONS = {
    "Home Dashboard": "🏠",
    "Flight Price Predictor": "✈️",
    "Trip & Hotel Recommendations": "🏨",
    "Gender Classification Demo": "👤",
    "Analytics": "📊",
}

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home Dashboard"


# ==============================================================================
# CUSTOM THEME & STYLING (NAVY BLUE / WHITE / MODERN BLUE ACCENTS)
# ==============================================================================


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Main Canvas */
        .stApp {
            background-color: #F5F8FC;
            color: #0F172A;
        }

        .block-container {
            padding-top: 3.1rem;
            padding-bottom: 2.5rem;
            max-width: 1360px;
        }

        header[data-testid="stHeader"] {
            background: rgba(245, 248, 252, 0.85);
            backdrop-filter: blur(6px);
        }

        /* Deep Navy Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #09182E 0%, #0F284B 55%, #12325C 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }

        [data-testid="stSidebar"] code {
            background: rgba(148, 163, 184, 0.18) !important;
            color: #93C5FD !important;
            padding: 0.1rem 0.4rem !important;
            border-radius: 5px !important;
            font-size: 0.72rem !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            text-align: left;
            justify-content: flex-start;
            border-radius: 10px;
            padding: 0.62rem 0.9rem;
            font-weight: 500;
            font-size: 0.93rem;
            border: 1px solid transparent;
            background: transparent;
            color: #CBD5E1;
            transition: all 0.18s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(56, 189, 248, 0.14);
            color: #FFFFFF;
            border-color: rgba(56, 189, 248, 0.35);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #1D6FE2 0%, #2563EB 100%);
            color: #FFFFFF !important;
            border-color: rgba(147, 197, 253, 0.4);
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(29, 111, 226, 0.35);
        }

        /* Sidebar Status Pills */
        .status-box {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 0.85rem 0.9rem;
            margin-top: 0.5rem;
        }
        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            margin: 0.35rem 0;
        }
        .pill-online {
            background: rgba(16, 185, 129, 0.2);
            color: #6EE7B7 !important;
            border: 1px solid rgba(16, 185, 129, 0.45);
            border-radius: 999px;
            padding: 0.12rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .pill-offline {
            background: rgba(239, 68, 68, 0.2);
            color: #FCA5A5 !important;
            border: 1px solid rgba(239, 68, 68, 0.45);
            border-radius: 999px;
            padding: 0.12rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 700;
        }

        /* Page Header Banner */
        .tiq-header {
            background: linear-gradient(135deg, #0B1F3A 0%, #153B6E 60%, #1D6FE2 100%);
            color: #FFFFFF;
            padding: 1.75rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(11, 31, 58, 0.12);
        }
        .tiq-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.22);
            color: #BAE6FD;
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 999px;
            padding: 0.2rem 0.75rem;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .tiq-header h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
        }
        .tiq-header p {
            margin: 0.45rem 0 0 0;
            font-size: 0.96rem;
            color: #DBEAFE;
            max-width: 820px;
            line-height: 1.5;
        }

        /* KPI Cards */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 4px solid #1D6FE2;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            height: 100%;
        }
        .kpi-card.teal { border-top-color: #0EA5A4; }
        .kpi-card.navy { border-top-color: #0B1F3A; }
        .kpi-card.indigo { border-top-color: #4F46E5; }
        .kpi-label {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: #0B1F3A;
            line-height: 1.15;
        }
        .kpi-sub {
            font-size: 0.8rem;
            color: #475569;
            margin-top: 0.3rem;
        }

        /* Content Card */
        .tiq-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }

        /* Hero Result Card */
        .price-hero {
            background: linear-gradient(135deg, #0B1F3A 0%, #1D6FE2 100%);
            color: #FFFFFF;
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 10px 26px rgba(29, 111, 226, 0.24);
        }
        .price-hero .label {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #BAE6FD;
        }
        .price-hero .amount {
            font-size: 2.65rem;
            font-weight: 800;
            margin: 0.25rem 0;
            letter-spacing: -0.02em;
            color: #FFFFFF;
        }
        .price-hero .meta {
            font-size: 0.85rem;
            color: #E0F2FE;
        }

        /* Package Recommendation Card */
        .pkg-card {
            background: #FFFFFF;
            border: 1px solid #DCE6F2;
            border-left: 5px solid #1D6FE2;
            border-radius: 14px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        .pkg-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .pkg-title {
            font-size: 1.06rem;
            font-weight: 700;
            color: #0B1F3A;
        }
        .pkg-score {
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            border-radius: 999px;
            padding: 0.22rem 0.72rem;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .pkg-costs {
            display: flex;
            flex-wrap: wrap;
            gap: 1.4rem;
            margin: 0.55rem 0;
            font-size: 0.88rem;
            color: #334155;
        }
        .pkg-costs strong {
            color: #0B1F3A;
        }
        .pkg-reason {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            font-size: 0.84rem;
            color: #334155;
            margin-top: 0.45rem;
        }

        /* Responsible AI / Transparency Callout */
        .notice-box {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 5px solid #D97706;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            color: #78350F;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-bottom: 1.2rem;
        }
        .info-box {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-left: 5px solid #1D6FE2;
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
            color: #1E3A8A;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-bottom: 1.1rem;
        }

        /* Primary CTA buttons in main content */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0F3D8A 0%, #1D6FE2 100%);
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.58rem 1.2rem;
            box-shadow: 0 4px 12px rgba(29, 111, 226, 0.25);
        }

        /* Hide default Streamlit hamburger/footer noise cleanly */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_logo_sidebar() -> None:
    """Render custom TRAVELIQ SVG logo inside the deep navy sidebar."""
    svg_light = """
    <div style="padding: 0.2rem 0 0.8rem 0;">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 245 58" width="218" height="52">
        <defs>
          <linearGradient id="sbGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#1D6FE2"/>
            <stop offset="100%" stop-color="#0EA5A4"/>
          </linearGradient>
        </defs>
        <rect x="2" y="4" width="48" height="48" rx="13" fill="url(#sbGrad)"/>
        <circle cx="26" cy="28" r="15.5" fill="none" stroke="#FFFFFF" stroke-width="1.6" stroke-opacity="0.45"/>
        <path d="M13 34 C19 20, 30 17, 40 21" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/>
        <polygon points="41,20 33,17 36,22 33,26" fill="#FFFFFF"/>
        <circle cx="15" cy="33" r="2.7" fill="#BAE6FD"/>
        <text x="61" y="30" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="23" font-weight="800" letter-spacing="1.1" fill="#FFFFFF">TRAVEL<tspan fill="#38BDF8">IQ</tspan></text>
        <text x="62" y="45" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="9.5" font-weight="600" letter-spacing="2.1" fill="#94A3B8">TRAVEL INTELLIGENCE</text>
      </svg>
    </div>
    """
    st.markdown(svg_light, unsafe_allow_html=True)


def render_page_header(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="tiq-header">
            <div class="tiq-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtitle: str, accent: str = "") -> None:
    cls = f"kpi-card {accent}".strip()
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=24, t=74, b=28),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#0F172A", size=12),
        title=dict(
            font=dict(size=14.5, color="#0B1F3A", family="Inter, Segoe UI, sans-serif"),
            x=0.01,
            y=0.96,
            xanchor="left",
            yanchor="top",
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )
    return fig


# ==============================================================================
# CACHED DATA & ENGINE LOADING
# ==============================================================================


@st.cache_resource(show_spinner="Initializing TRAVELIQ datasets & recommendation engine...")
def load_platform_resources():
    flights_df = pd.read_csv(DATASET_DIR / "flights.csv")
    hotels_df = pd.read_csv(DATASET_DIR / "hotels.csv")
    users_df = pd.read_csv(DATASET_DIR / "users.csv")

    flights_df["date"] = pd.to_datetime(flights_df["date"], format="%m/%d/%Y")
    hotels_df["date"] = pd.to_datetime(hotels_df["date"], format="%m/%d/%Y")

    recommender = TravelPackageRecommender(
        flights_df=flights_df,
        hotels_df=hotels_df,
        users_df=users_df,
    )
    return flights_df, hotels_df, users_df, recommender


@st.cache_data(ttl=15, show_spinner=False)
def get_live_api_statuses() -> Dict[str, Dict[str, Any]]:
    return {
        "flight": check_api_health(FLIGHT_API_URL, timeout=2.5),
        "gender": check_api_health(GENDER_API_URL, timeout=2.5),
    }


def navigate_to(page_name: str) -> None:
    st.session_state["active_page"] = page_name


# ==============================================================================
# SIDEBAR NAVIGATION & LIVE SERVICE HEALTH
# ==============================================================================


def render_sidebar(api_statuses: Dict[str, Dict[str, Any]]) -> str:
    with st.sidebar:
        render_brand_logo_sidebar()
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;"
            "color:#94A3B8;text-transform:uppercase;margin:0.3rem 0 0.5rem 0.2rem;'>"
            "Navigation</div>",
            unsafe_allow_html=True,
        )

        for page in NAV_PAGES:
            is_active = st.session_state["active_page"] == page
            if st.button(
                f"{NAV_ICONS[page]}  {page}",
                key=f"nav_{page}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["active_page"] = page
                st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,0.12);margin:1.1rem 0;'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;"
            "color:#94A3B8;text-transform:uppercase;margin-bottom:0.35rem;'>"
            "Live API Services</div>",
            unsafe_allow_html=True,
        )

        f_stat = api_statuses["flight"]
        g_stat = api_statuses["gender"]

        f_pill = "<span class='pill-online'>ONLINE</span>" if f_stat["online"] else "<span class='pill-offline'>OFFLINE</span>"
        g_pill = "<span class='pill-online'>ONLINE</span>" if g_stat["online"] else "<span class='pill-offline'>OFFLINE</span>"

        f_src = f_stat["details"].get("model_source", "unavailable") if f_stat["online"] else "not connected"
        g_ver = g_stat["details"].get("artifact_version", "unavailable") if g_stat["online"] else "not connected"

        st.markdown(
            f"""
            <div class="status-box">
                <div class="status-row">
                    <span>✈️ Flight API</span>
                    {f_pill}
                </div>
                <div style="font-size:0.72rem;color:#94A3B8;margin-bottom:0.55rem;">
                    Source: {f_src}
                </div>
                <div class="status-row">
                    <span>👤 Gender API</span>
                    {g_pill}
                </div>
                <div style="font-size:0.72rem;color:#94A3B8;">
                    Model: {g_ver}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh API Health", use_container_width=True):
            get_live_api_statuses.clear()
            st.rerun()

        st.markdown(
            "<div style='font-size:0.72rem;color:#64748B;margin-top:1.2rem;line-height:1.45;'>"
            "TRAVELIQ v2.0 · Travel Intelligence Platform<br>"
            f"Flight API: <code>{FLIGHT_API_URL}</code><br>"
            f"Gender API: <code>{GENDER_API_URL}</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    return st.session_state["active_page"]


# ==============================================================================
# SECTION 1: HOME DASHBOARD
# ==============================================================================


def render_home_dashboard(
    flights: pd.DataFrame,
    hotels: pd.DataFrame,
    users: pd.DataFrame,
    recommender: TravelPackageRecommender,
    api_statuses: Dict[str, Dict[str, Any]],
) -> None:
    render_page_header(
        badge="TRAVEL INTELLIGENCE PLATFORM",
        title="Welcome to TRAVELIQ",
        subtitle=(
            "An end-to-end machine learning platform uniting flight fare forecasting, "
            "hybrid flight-and-hotel package recommendations, experimental traveler profiling, "
            "and interactive market analytics across Brazilian corporate travel records."
        ),
    )

    # Top KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card(
            "Total Flights",
            f"{len(flights):,}",
            f"{flights.groupby(['from', 'to']).ngroups} active routes · 3 agencies",
            accent="",
        )
    with k2:
        render_kpi_card(
            "Hotel Stays",
            f"{len(hotels):,}",
            f"{hotels['name'].nunique()} partner hotels (1 per city)",
            accent="teal",
        )
    with k3:
        render_kpi_card(
            "Traveler Profiles",
            f"{len(users):,}",
            f"{users['company'].nunique()} corporate accounts",
            accent="navy",
        )
    with k4:
        render_kpi_card(
            "Destinations",
            f"{hotels['place'].nunique()}",
            "Brazilian regional hub cities",
            accent="indigo",
        )
    with k5:
        render_kpi_card(
            "Travel Packages",
            f"{len(recommender.package_catalog):,}",
            "Combined round-trip + hotel offers",
            accent="teal",
        )

    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

    # Quick Navigation Launchpad
    st.markdown("### Quick Launch")
    q1, q2, q3, q4 = st.columns(4)
    quick_items = [
        (
            q1,
            "✈️ Flight Price Predictor",
            "Estimate airfare via the live MLflow-backed LightGBM REST API.",
            "Flight Price Predictor",
            "Open Predictor →",
        ),
        (
            q2,
            "🏨 Trip & Hotel Packages",
            "Rank 1,568 round-trip flight + hotel bundles with budget & CF affinity.",
            "Trip & Hotel Recommendations",
            "Explore Packages →",
        ),
        (
            q3,
            "👤 Gender Classification",
            "Test the lightweight 40.4 KB binary classifier via live Gender API.",
            "Gender Classification Demo",
            "Open Demo →",
        ),
        (
            q4,
            "📊 Travel Analytics",
            "Inspect fare distributions, popular routes, stay lengths & costs.",
            "Analytics",
            "View Analytics →",
        ),
    ]
    for col, title, desc, target, btn_label in quick_items:
        with col:
            st.markdown(
                f"""
                <div class="tiq-card" style="min-height:132px;margin-bottom:0.55rem;">
                    <div style="font-weight:700;font-size:1rem;color:#0B1F3A;margin-bottom:0.35rem;">{title}</div>
                    <div style="font-size:0.84rem;color:#475569;line-height:1.45;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                btn_label,
                key=f"quick_{target}",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=(target,),
            )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Overview Analytics Charts
    st.markdown("### Platform Travel Overview")
    c1, c2 = st.columns(2)

    with c1:
        class_agency = (
            flights.groupby(["flightType", "agency"], as_index=False)["price"]
            .mean()
            .rename(columns={"price": "Avg Price ($)"})
        )
        fig_ca = px.bar(
            class_agency,
            x="flightType",
            y="Avg Price ($)",
            color="agency",
            barmode="group",
            category_orders={"flightType": ["economic", "premium", "firstClass"]},
            color_discrete_sequence=["#0B1F3A", "#1D6FE2", "#0EA5A4"],
            title="Average Flight Fare by Cabin Class & Agency",
        )
        fig_ca.update_xaxes(title="Cabin Class")
        fig_ca.update_yaxes(title="Average Ticket Price ($)")
        st.plotly_chart(style_plotly(fig_ca, height=370), use_container_width=True)

    with c2:
        dest_pop = recommender.city_hotel_summary.sort_values("bookings", ascending=True)
        fig_dp = px.bar(
            dest_pop,
            x="bookings",
            y="destination",
            orientation="h",
            color="price_per_day",
            color_continuous_scale=["#93C5FD", "#1D6FE2", "#0B1F3A"],
            hover_data=["hotel_name", "price_per_day", "unique_users"],
            title="Destination Demand & Partner Hotel Daily Rate",
        )
        fig_dp.update_xaxes(title="Historical Hotel Bookings")
        fig_dp.update_yaxes(title="")
        fig_dp.update_coloraxes(colorbar_title="$/Day")
        st.plotly_chart(style_plotly(fig_dp, height=370), use_container_width=True)

    # Dataset Summary Expander
    with st.expander("📋 Dataset Coverage & Architecture Notes", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f"**`flights.csv`**\n"
                f"- Records: **{len(flights):,}**\n"
                f"- Date Range: `{flights['date'].min().date()}` to `{flights['date'].max().date()}`\n"
                f"- Price Range: **${flights['price'].min():,.2f} – ${flights['price'].max():,.2f}**\n"
                f"- Flight Time: **{flights['time'].min():.2f} – {flights['time'].max():.2f} hrs**"
            )
        with col_b:
            st.markdown(
                f"**`hotels.csv`**\n"
                f"- Records: **{len(hotels):,}**\n"
                f"- Cities / Hotels: **9 destinations ↔ 9 partner hotels**\n"
                f"- Daily Rate: **${hotels['price'].min():,.2f} – ${hotels['price'].max():,.2f}**\n"
                f"- Stay Durations: **1 to 4 days**"
            )
        with col_c:
            labeled_n = int(users["gender"].isin(["male", "female"]).sum())
            none_n = int((users["gender"] == "none").sum())
            st.markdown(
                f"**`users.csv`**\n"
                f"- Records: **{len(users):,}**\n"
                f"- Labeled Gender (`male`/`female`): **{labeled_n:,}**\n"
                f"- Undisclosed (`none`): **{none_n:,}** (held out from training)\n"
                f"- Age Range: **{int(users['age'].min())} – {int(users['age'].max())} yrs**"
            )


# ==============================================================================
# SECTION 2: FLIGHT PRICE PREDICTOR
# ==============================================================================


def render_flight_predictor(
    flights: pd.DataFrame,
    api_statuses: Dict[str, Dict[str, Any]],
) -> None:
    render_page_header(
        badge="MLFLOW-MANAGED REGRESSION",
        title="Flight Price Predictor",
        subtitle=(
            "Configure route, cabin class, agency, travel date, duration, and distance. "
            "Predictions are served live from the TRAVELIQ Flask Flight API backed by the "
            "LightGBM regression pipeline."
        ),
    )

    if not api_statuses["flight"]["online"]:
        st.warning(
            f"Flight Price API ({FLIGHT_API_URL}) is currently offline. "
            "Start it with `python -m app.app` to run live predictions."
        )

    left_col, right_col = st.columns([1.15, 1.0], gap="large")

    with left_col:
        st.markdown("#### Configure Flight Parameters")

        origins = sorted(flights["from"].dropna().astype(str).unique())
        default_from = "Florianopolis (SC)" if "Florianopolis (SC)" in origins else origins[0]
        flight_from = st.selectbox("Origin City", origins, index=origins.index(default_from))

        route_from_df = flights[flights["from"] == flight_from]
        destinations = sorted(route_from_df["to"].dropna().astype(str).unique())
        default_to = "Aracaju (SE)" if "Aracaju (SE)" in destinations else destinations[0]
        flight_to = st.selectbox("Destination City", destinations, index=destinations.index(default_to))

        selected_route = route_from_df[route_from_df["to"] == flight_to]

        c_type, c_agency = st.columns(2)
        with c_type:
            type_options = sorted(selected_route["flightType"].dropna().astype(str).unique())
            default_type = "firstClass" if "firstClass" in type_options else type_options[0]
            flight_type = st.selectbox("Cabin Class (Flight Type)", type_options, index=type_options.index(default_type))

        with c_agency:
            agency_df = selected_route[selected_route["flightType"] == flight_type]
            agency_options = sorted(agency_df["agency"].dropna().astype(str).unique())
            default_agency = "CloudFy" if "CloudFy" in agency_options else agency_options[0]
            agency = st.selectbox("Airline / Agency", agency_options, index=agency_options.index(default_agency))

        combo_df = agency_df[agency_df["agency"] == agency]
        if combo_df.empty:
            combo_df = selected_route

        default_time = round(float(combo_df["time"].median()), 2)
        default_distance = round(float(combo_df["distance"].median()), 2)

        c_time, c_dist, c_date = st.columns(3)
        with c_time:
            flight_time = st.number_input(
                "Flight Duration (hours)",
                min_value=0.10,
                max_value=24.0,
                value=default_time,
                step=0.05,
                format="%.2f",
                help="Dataset flight times range from 0.44 to 2.44 hours.",
            )
        with c_dist:
            distance = st.number_input(
                "Distance (km)",
                min_value=10.0,
                max_value=15000.0,
                value=default_distance,
                step=10.0,
                format="%.2f",
            )
        with c_date:
            travel_date = st.date_input("Travel Date", value=date(2026, 9, 20))

        predict_clicked = st.button(
            "💰 Predict Flight Price",
            type="primary",
            use_container_width=True,
        )

    with right_col:
        st.markdown("#### Prediction & Model Output")

        if predict_clicked:
            payload = {
                "from": flight_from,
                "to": flight_to,
                "flightType": flight_type,
                "agency": agency,
                "time": float(flight_time),
                "distance": float(distance),
                "date": travel_date.strftime("%Y-%m-%d"),
            }
            with st.spinner("Contacting TRAVELIQ Flight Price API..."):
                result, error = request_flight_prediction(payload)

            if error:
                st.error(f"Prediction Error: {error}")
            else:
                st.session_state["last_flight_result"] = result
                st.session_state["last_flight_payload"] = payload
                st.session_state["last_route_avg"] = float(combo_df["price"].mean())

        if "last_flight_result" in st.session_state:
            res = st.session_state["last_flight_result"]
            pay = st.session_state["last_flight_payload"]
            hist_avg = st.session_state.get("last_route_avg", 0.0)
            pred_price = float(res["predicted_price"])
            diff = pred_price - hist_avg

            st.markdown(
                f"""
                <div class="price-hero">
                    <div class="label">Estimated One-Way Flight Fare</div>
                    <div class="amount">${pred_price:,.2f} <span style="font-size:1rem;font-weight:600;opacity:0.85;">USD</span></div>
                    <div class="meta">
                        Route: <strong>{pay['from']} → {pay['to']}</strong> · {pay['flightType']} · {pay['agency']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                render_kpi_card(
                    "Historical Route Mean",
                    f"${hist_avg:,.2f}",
                    f"Delta vs prediction: {diff:+.2f} USD",
                    accent="teal",
                )
            with m2:
                src_label = (
                    "MLflow @champion"
                    if res.get("model_source") == "mlflow_registry"
                    else "Local Joblib Fallback"
                )
                render_kpi_card(
                    "Model Serving Source",
                    src_label,
                    f"{res.get('model', 'Voyage_Flight_Price_Model')} ({res.get('model_alias', 'champion')})",
                    accent="navy",
                )

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            with st.expander("🧾 Request Summary & Model Details", expanded=True):
                summary_df = pd.DataFrame(
                    [
                        {"Field": "Origin", "Value": pay["from"]},
                        {"Field": "Destination", "Value": pay["to"]},
                        {"Field": "Cabin Class", "Value": pay["flightType"]},
                        {"Field": "Agency", "Value": pay["agency"]},
                        {"Field": "Duration", "Value": f"{pay['time']:.2f} hours"},
                        {"Field": "Distance", "Value": f"{pay['distance']:.2f} km"},
                        {"Field": "Travel Date", "Value": pay["date"]},
                        {"Field": "API Endpoint", "Value": f"{FLIGHT_API_URL}/predict"},
                    ]
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                """
                <div class="tiq-card" style="text-align:center;padding:2.6rem 1.5rem;">
                    <div style="font-size:2.2rem;margin-bottom:0.5rem;">✈️</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#0B1F3A;">Ready for Fare Estimation</div>
                    <div style="font-size:0.88rem;color:#64748B;margin-top:0.35rem;">
                        Select your route, cabin class, agency, and date on the left, then click
                        <strong>Predict Flight Price</strong> to query the live Flask API.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# SECTION 3: TRIP & HOTEL RECOMMENDATIONS
# ==============================================================================


def render_trip_recommendations(recommender: TravelPackageRecommender) -> None:
    render_page_header(
        badge="HYBRID COLLABORATIVE + VALUE ENGINE",
        title="Trip & Hotel Package Recommendations",
        subtitle=(
            "Discover ranked round-trip Flight + Hotel travel packages tailored to your departure city, "
            "trip duration, total budget, and optional traveler booking history."
        ),
    )

    # Clear transparent communication of the 1-hotel-per-city dataset limitation
    st.markdown(
        f"""
        <div class="info-box">
            <strong>ℹ️ Important Dataset Limitation — One Partner Hotel per City:</strong><br>
            {ONE_HOTEL_LIMITATION_NOTE}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter controls
    c1, c2, c3 = st.columns(3)
    with c1:
        origin_choice = st.selectbox(
            "Departure Origin City",
            ["Any Origin (or Traveler's Hub)"] + recommender.valid_cities,
        )
    with c2:
        dest_choice = st.selectbox(
            "Destination Preference",
            ["Explore All Destinations"] + recommender.valid_cities,
        )
    with c3:
        class_choice = st.selectbox(
            "Preferred Cabin Class",
            ["Any Cabin Class"] + recommender.valid_flight_types,
        )

    c4, c5, c6, c7 = st.columns([1, 1.3, 1, 1.4])
    with c4:
        days = st.selectbox("Trip Duration (Days)", [1, 2, 3, 4], index=2)
    with c5:
        max_budget = st.number_input(
            "Total Package Budget ($ USD)",
            min_value=100.0,
            max_value=6000.0,
            value=2500.0,
            step=100.0,
            help="Covers BOTH round-trip flight AND hotel stay.",
        )
    with c6:
        top_n = st.slider("Packages to Show", min_value=1, max_value=10, value=5)
    with c7:
        user_options = ["Anonymous / Cold-Start"] + [
            f"User #{u} — {recommender.user_profiles.loc[u, 'name']} ({recommender.user_profiles.loc[u, 'company']})"
            for u in recommender.known_user_codes[:100]
        ]
        selected_user_label = st.selectbox(
            "Personalize for Traveler (Optional)",
            user_options,
            help="Uses historical hotel stays (User-Based CF) and cabin preferences.",
        )

    opt_c1, opt_c2 = st.columns(2)
    with opt_c1:
        strict_days = st.checkbox("Require exact trip duration match", value=False)
    with opt_c2:
        diverse_dests = st.checkbox(
            "Prioritize diverse destinations (when exploring)", value=True
        )

    selected_user_code = None
    if selected_user_label != "Anonymous / Cold-Start":
        selected_user_code = int(selected_user_label.split("#")[1].split(" ")[0])
        profile = recommender.get_user_profile(selected_user_code)
        if profile:
            st.markdown(
                f"""
                <div class="tiq-card" style="padding:0.85rem 1.15rem;background:#F8FAFC;">
                    <strong>Traveler Profile:</strong> {profile['name']} ({profile['company']}) ·
                    <strong>{profile['total_trips']}</strong> historical trips ·
                    Typical Hub: <strong>{profile['dominant_origin']}</strong> ·
                    Preferred Cabin: <strong>{profile['preferred_flight_type']}</strong> ·
                    Avg Spend: <strong>${profile['avg_package_spend']:,.2f}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    origin_arg = None if origin_choice.startswith("Any Origin") else origin_choice
    dest_arg = None if dest_choice == "Explore All Destinations" else dest_choice
    class_arg = None if class_choice == "Any Cabin Class" else class_choice

    recs_df, meta = recommender.recommend(
        days=int(days),
        max_budget=float(max_budget),
        origin=origin_arg,
        destination=dest_arg,
        flight_type=class_arg,
        user_code=selected_user_code,
        top_n=int(top_n),
        strict_days=strict_days,
        diverse_destinations=diverse_dests,
    )

    if meta["status"] == "error":
        st.error(meta["message"])
        return

    if meta["status"] == "no_match" or recs_df.empty:
        st.warning(meta["message"])
        if meta.get("notes"):
            for note in meta["notes"]:
                st.caption(f"• {note}")
        return

    # Notes & summary KPIs
    if meta.get("notes"):
        st.info("  \n".join(f"• {note}" for note in meta["notes"]))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Top Match Score",
            f"{recs_df['recommendation_score'].iloc[0]:.1f} / 100",
            f"{'Personalized CF + Value' if meta['personalized'] else 'Budget & Popularity'}",
            accent="",
        )
    with k2:
        render_kpi_card(
            "Lowest Package Cost",
            f"${recs_df['package_total'].min():,.2f}",
            "Round-trip flight + hotel combined",
            accent="teal",
        )
    with k3:
        render_kpi_card(
            "Avg Budget Savings",
            f"${recs_df['budget_savings'].mean():,.2f}",
            f"Under ${max_budget:,.0f} ceiling",
            accent="navy",
        )
    with k4:
        render_kpi_card(
            "Eligible Matches",
            f"{meta['total_eligible_packages']:,}",
            f"Showing Top {meta['returned_count']} across {recs_df['destination'].nunique()} city/cities",
            accent="indigo",
        )

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

    # Stacked Cost Breakdown Chart (Flight vs Hotel separately)
    chart_df = recs_df.copy()
    chart_df["pkg_label"] = (
        "#" + chart_df["rank"].astype(str) + " "
        + chart_df["destination"] + " (" + chart_df["hotel_name"] + " · " + chart_df["flightType"] + ")"
    )
    fig_pkg = go.Figure()
    fig_pkg.add_trace(
        go.Bar(
            y=chart_df["pkg_label"],
            x=chart_df["flight_roundtrip_price"],
            name="Round-Trip Flight Cost ($)",
            orientation="h",
            marker_color="#1D6FE2",
            hovertemplate="Flight Cost: $%{x:,.2f}<extra></extra>",
        )
    )
    fig_pkg.add_trace(
        go.Bar(
            y=chart_df["pkg_label"],
            x=chart_df["hotel_total"],
            name="Hotel Stay Cost ($)",
            orientation="h",
            marker_color="#0EA5A4",
            hovertemplate="Hotel Cost: $%{x:,.2f}<extra></extra>",
        )
    )
    fig_pkg.add_vline(
        x=max_budget,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Budget ${max_budget:,.0f}",
        annotation_position="top right",
    )
    fig_pkg.update_layout(
        barmode="stack",
        title="Recommended Package Cost Breakdown: Round-Trip Flight vs. Hotel Stay",
        xaxis_title="Total Cost ($ USD)",
        yaxis=dict(autorange="reversed", title=""),
    )
    st.plotly_chart(style_plotly(fig_pkg, height=max(320, 56 * len(chart_df) + 110)), use_container_width=True)

    # Detailed Ranked Package Cards
    st.markdown("#### Ranked Travel Packages & Recommendation Rationale")
    for _, row in recs_df.iterrows():
        st.markdown(
            f"""
            <div class="pkg-card">
                <div class="pkg-top">
                    <div class="pkg-title">
                        #{int(row['rank'])} &nbsp; {row['origin']} → {row['destination']}
                        &nbsp;·&nbsp; <span style="color:#1D6FE2;">{row['hotel_name']}</span>
                        &nbsp;({int(row['days'])} {'day' if int(row['days']) == 1 else 'days'} · {row['flightType']} · {row['agency']})
                    </div>
                    <div class="pkg-score">Score {row['recommendation_score']:.1f} / 100</div>
                </div>
                <div class="pkg-costs">
                    <span>✈️ Round-Trip Flight: <strong>${row['flight_roundtrip_price']:,.2f}</strong></span>
                    <span>🏨 Hotel ({int(row['days'])}d × ${row['hotel_price_per_day']:,.2f}): <strong>${row['hotel_total']:,.2f}</strong></span>
                    <span>💳 Total Package: <strong style="color:#1D6FE2;">${row['package_total']:,.2f}</strong></span>
                    <span>💰 Savings: <strong style="color:#059669;">${row['budget_savings']:,.2f}</strong></span>
                </div>
                <div class="pkg-reason">
                    <strong>Why recommended:</strong> {row['recommendation_reason']}
                    <span style="color:#64748B;">
                        &nbsp;[Budget Fit: {row['budget_score']*100:.0f}% · Duration: {row['duration_score']*100:.0f}% ·
                        Personal/CF: {row['personal_score']*100:.0f}% · Popularity: {row['popularity_score']*100:.0f}%]
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full comparison table
    with st.expander("📄 View Full Tabular Package Breakdown", expanded=False):
        table_df = recs_df[
            [
                "rank",
                "origin",
                "destination",
                "hotel_name",
                "days",
                "flightType",
                "agency",
                "flight_roundtrip_price",
                "hotel_total",
                "package_total",
                "recommendation_score",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "origin": "Origin",
                "destination": "Destination",
                "hotel_name": "Hotel",
                "days": "Days",
                "flightType": "Cabin Class",
                "agency": "Agency",
                "flight_roundtrip_price": "Flight Cost ($)",
                "hotel_total": "Hotel Cost ($)",
                "package_total": "Total Cost ($)",
                "recommendation_score": "Score (/100)",
            }
        )
        st.dataframe(
            table_df.style.format(
                {
                    "Flight Cost ($)": "${:,.2f}",
                    "Hotel Cost ($)": "${:,.2f}",
                    "Total Cost ($)": "${:,.2f}",
                    "Score (/100)": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# SECTION 4: GENDER CLASSIFICATION DEMO
# ==============================================================================


def render_gender_demo(
    users: pd.DataFrame,
    api_statuses: Dict[str, Dict[str, Any]],
) -> None:
    render_page_header(
        badge="EXPERIMENTAL ML MICROSERVICE",
        title="Gender Classification Demo",
        subtitle=(
            "Interactive demonstration of the lightweight 40.4 KB scikit-learn binary classifier "
            "served via the TRAVELIQ Gender Classification REST API."
        ),
    )

    st.markdown(
        f"""
        <div class="notice-box">
            <strong>⚠️ Responsible AI & Experimental Usage Notice:</strong><br>
            {EXPERIMENTAL_GENDER_NOTICE}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_statuses["gender"]["online"]:
        st.warning(
            f"Gender Classification API ({GENDER_API_URL}) is currently offline. "
            "Start it with `python -m app.gender_api` to run live classification."
        )

    left_col, right_col = st.columns([1.05, 1.0], gap="large")

    with left_col:
        st.markdown("#### Traveler Input Fields")
        companies = sorted(users["company"].dropna().astype(str).unique().tolist())

        name_input = st.text_input(
            "Traveler Full Name",
            value="Wilma Mcinnis",
            placeholder="e.g., John Smith or Priya Sharma",
        )
        company_input = st.selectbox("Company", companies)
        age_input = st.number_input("Age (Years)", min_value=18, max_value=100, value=38, step=1)

        classify_clicked = st.button(
            "🔍 Run Experimental Classification",
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "Model Note: Trained on 900 binary-labeled records from `users.csv` "
            "(440 `'none'` undisclosed rows excluded from supervised training). "
            "Holdout accuracy: 88.33% · 5-Fold CV F1: 90.83% · ROC-AUC: 0.9533."
        )

    with right_col:
        st.markdown("#### Experimental Model Output")

        if classify_clicked:
            if not name_input.strip():
                st.error("Please enter a non-empty traveler name.")
            else:
                payload = {
                    "name": name_input.strip(),
                    "company": company_input,
                    "age": int(age_input),
                }
                with st.spinner("Contacting TRAVELIQ Gender Classification API..."):
                    result, error = request_gender_prediction(payload)

                if error:
                    st.error(f"Classification Error: {error}")
                else:
                    st.session_state["last_gender_result"] = result
                    st.session_state["last_gender_payload"] = payload

        if "last_gender_result" in st.session_state:
            gres = st.session_state["last_gender_result"]
            gpay = st.session_state["last_gender_payload"]
            pred_label = str(gres["predicted_gender"]).title()
            conf = float(gres.get("confidence", 0.0))
            probs = gres.get("probabilities", {})

            st.markdown(
                f"""
                <div class="price-hero" style="background:linear-gradient(135deg,#0B1F3A 0%,#2563EB 100%);">
                    <div class="label">Experimental Statistical Estimate (Not Verified Identity)</div>
                    <div class="amount" style="font-size:2.1rem;">{pred_label}</div>
                    <div class="meta">
                        Model Probability Confidence: <strong>{conf * 100:.1f}%</strong> ·
                        Input: {gpay['name']} ({gpay['company']}, age {gpay['age']})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if probs:
                prob_df = pd.DataFrame(
                    [
                        {"Class": k.title(), "Probability (%)": round(float(v) * 100, 2)}
                        for k, v in probs.items()
                    ]
                )
                fig_prob = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Class",
                    orientation="h",
                    range_x=[0, 120],
                    color="Class",
                    color_discrete_map={"Female": "#0EA5A4", "Male": "#1D6FE2"},
                    text="Probability (%)",
                    title="Calibrated Class Probability Breakdown",
                )
                fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(style_plotly(fig_prob, height=235), use_container_width=True)

            st.caption(
                f"API: `{GENDER_API_URL}/predict-gender` · "
                f"Artifact: `{gres.get('artifact_version', '2.0-lightweight-binary')}`"
            )
        else:
            st.markdown(
                """
                <div class="tiq-card" style="text-align:center;padding:2.6rem 1.5rem;">
                    <div style="font-size:2.2rem;margin-bottom:0.5rem;">👤</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#0B1F3A;">Ready for Experimental Demo</div>
                    <div style="font-size:0.88rem;color:#64748B;margin-top:0.35rem;">
                        Enter a name, company, and age on the left to inspect the lightweight
                        binary classifier's statistical estimate and calibrated confidence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# SECTION 5: ANALYTICS
# ==============================================================================


def render_analytics(
    flights: pd.DataFrame,
    hotels: pd.DataFrame,
    recommender: TravelPackageRecommender,
) -> None:
    render_page_header(
        badge="EXPLORATORY MARKET INTELLIGENCE",
        title="Travel & Pricing Analytics",
        subtitle=(
            "Interactive analysis of flight fare distributions, top corridors, destination demand, "
            "trip durations, and combined Flight + Hotel package pricing."
        ),
    )

    tab_fare, tab_routes, tab_dest, tab_dur, tab_pkg = st.tabs(
        [
            "💰 Flight Price Distribution",
            "🗺️ Popular Routes",
            "🏙️ Destination Popularity",
            "⏱️ Travel Duration",
            "🏨 Hotel & Package Pricing",
        ]
    )

    # 1. Flight Price Distribution
    with tab_fare:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                flights,
                x="price",
                nbins=45,
                color="flightType",
                barmode="overlay",
                opacity=0.72,
                category_orders={"flightType": ["economic", "premium", "firstClass"]},
                color_discrete_sequence=["#0EA5A4", "#1D6FE2", "#0B1F3A"],
                title="Flight Fare Distribution by Cabin Class",
            )
            fig_hist.update_xaxes(title="One-Way Flight Price ($)")
            fig_hist.update_yaxes(title="Flight Count")
            st.plotly_chart(style_plotly(fig_hist, height=400), use_container_width=True)

        with c2:
            sample_f = flights.sample(n=min(15000, len(flights)), random_state=42)
            fig_box = px.box(
                sample_f,
                x="agency",
                y="price",
                color="flightType",
                category_orders={"flightType": ["economic", "premium", "firstClass"]},
                color_discrete_sequence=["#0EA5A4", "#1D6FE2", "#0B1F3A"],
                title="Fare Spread by Travel Agency & Cabin Class",
            )
            fig_box.update_xaxes(title="Agency")
            fig_box.update_yaxes(title="Price ($)")
            st.plotly_chart(style_plotly(fig_box, height=400), use_container_width=True)

    # 2. Popular Routes
    with tab_routes:
        route_stats = (
            flights.assign(route=flights["from"] + " → " + flights["to"])
            .groupby("route", as_index=False)
            .agg(
                bookings=("price", "count"),
                avg_price=("price", "mean"),
                avg_distance=("distance", "mean"),
                avg_time=("time", "mean"),
            )
        )
        c1, c2 = st.columns(2)
        with c1:
            top_vol = route_stats.nlargest(10, "bookings").sort_values("bookings")
            fig_rv = px.bar(
                top_vol,
                x="bookings",
                y="route",
                orientation="h",
                color="avg_price",
                color_continuous_scale=["#60A5FA", "#1D6FE2", "#0B1F3A"],
                title="Top 10 Busiest Flight Corridors",
            )
            fig_rv.update_xaxes(title="Total Flight Legs")
            fig_rv.update_yaxes(title="")
            st.plotly_chart(style_plotly(fig_rv, height=420), use_container_width=True)

        with c2:
            top_exp = route_stats.nlargest(10, "avg_price").sort_values("avg_price")
            fig_re = px.bar(
                top_exp,
                x="avg_price",
                y="route",
                orientation="h",
                color="avg_distance",
                color_continuous_scale=["#5EEAD4", "#0EA5A4", "#0F766E"],
                title="Top 10 Highest-Fare Flight Routes",
            )
            fig_re.update_xaxes(title="Average One-Way Fare ($)")
            fig_re.update_yaxes(title="")
            st.plotly_chart(style_plotly(fig_re, height=420), use_container_width=True)

    # 3. Destination Popularity
    with tab_dest:
        dest_df = recommender.city_hotel_summary.copy()
        c1, c2 = st.columns(2)
        with c1:
            fig_d1 = px.bar(
                dest_df.sort_values("bookings"),
                x="bookings",
                y="destination",
                orientation="h",
                color="unique_users",
                color_continuous_scale=["#93C5FD", "#1D6FE2", "#0B1F3A"],
                hover_data=["hotel_name", "price_per_day"],
                title="Total Hotel Stays by Destination",
            )
            fig_d1.update_xaxes(title="Total Hotel Bookings")
            fig_d1.update_yaxes(title="")
            st.plotly_chart(style_plotly(fig_d1, height=400), use_container_width=True)

        with c2:
            fig_d2 = px.scatter(
                dest_df,
                x="price_per_day",
                y="bookings",
                size="unique_users",
                color="destination",
                text="hotel_name",
                title="Destination Demand vs. Partner Hotel Daily Rate",
            )
            fig_d2.update_traces(textposition="top center")
            fig_d2.update_xaxes(title="Fixed Hotel Daily Rate ($)")
            fig_d2.update_yaxes(title="Total Bookings")
            fig_d2.update_layout(showlegend=False)
            st.plotly_chart(style_plotly(fig_d2, height=400), use_container_width=True)

    # 4. Travel Duration
    with tab_dur:
        c1, c2 = st.columns(2)
        with c1:
            fig_ft = px.histogram(
                flights,
                x="time",
                nbins=30,
                color_discrete_sequence=["#1D6FE2"],
                title="Flight Duration Distribution (Hours per Leg)",
            )
            fig_ft.update_xaxes(title="Flight Time (Hours)")
            fig_ft.update_yaxes(title="Number of Flights")
            st.plotly_chart(style_plotly(fig_ft, height=380), use_container_width=True)

        with c2:
            stay_counts = hotels["days"].value_counts().sort_index().reset_index()
            stay_counts.columns = ["Stay Duration (Days)", "Bookings"]
            fig_hd = px.bar(
                stay_counts,
                x="Stay Duration (Days)",
                y="Bookings",
                text="Bookings",
                color_discrete_sequence=["#0EA5A4"],
                title="Hotel Stay Length Distribution (1–4 Days)",
            )
            fig_hd.update_traces(texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(style_plotly(fig_hd, height=380), use_container_width=True)

    # 5. Hotel & Package Price Analysis
    with tab_pkg:
        trips = recommender.trips
        c1, c2 = st.columns(2)
        with c1:
            pkg_class = (
                trips.groupby("flightType", as_index=False)[["flight_roundtrip_price", "total"]]
                .mean()
                .rename(
                    columns={
                        "flight_roundtrip_price": "Round-Trip Flight ($)",
                        "total": "Hotel Stay ($)",
                    }
                )
            )
            pkg_melt = pkg_class.melt(
                id_vars="flightType",
                value_vars=["Round-Trip Flight ($)", "Hotel Stay ($)"],
                var_name="Cost Component",
                value_name="Average Cost ($)",
            )
            fig_pc = px.bar(
                pkg_melt,
                x="flightType",
                y="Average Cost ($)",
                color="Cost Component",
                barmode="stack",
                category_orders={"flightType": ["economic", "premium", "firstClass"]},
                color_discrete_sequence=["#1D6FE2", "#0EA5A4"],
                title="Average Full Package Cost: Round-Trip Flight + Hotel",
            )
            fig_pc.update_xaxes(title="Cabin Class")
            st.plotly_chart(style_plotly(fig_pc, height=400), use_container_width=True)

        with c2:
            dest_pkg = (
                trips.groupby("place", as_index=False)["package_total"]
                .agg(mean_pkg="mean", min_pkg="min", max_pkg="max")
                .sort_values("mean_pkg")
            )
            fig_dp2 = px.bar(
                dest_pkg,
                x="mean_pkg",
                y="place",
                orientation="h",
                color="mean_pkg",
                color_continuous_scale=["#93C5FD", "#1D6FE2", "#0B1F3A"],
                hover_data={"min_pkg": ":$.2f", "max_pkg": ":$.2f"},
                title="Average Total Travel Package Cost by Destination",
            )
            fig_dp2.update_xaxes(title="Average Full Package Cost ($)")
            fig_dp2.update_yaxes(title="")
            st.plotly_chart(style_plotly(fig_dp2, height=400), use_container_width=True)


# ==============================================================================
# APPLICATION ENTRYPOINT
# ==============================================================================


def main() -> None:
    inject_custom_css()

    try:
        flights, hotels, users, recommender = load_platform_resources()
    except Exception as exc:
        st.error(f"Failed to load TRAVELIQ datasets from `{DATASET_DIR}`: {exc}")
        st.stop()

    api_statuses = get_live_api_statuses()
    active_page = render_sidebar(api_statuses)

    if active_page == "Home Dashboard":
        render_home_dashboard(flights, hotels, users, recommender, api_statuses)
    elif active_page == "Flight Price Predictor":
        render_flight_predictor(flights, api_statuses)
    elif active_page == "Trip & Hotel Recommendations":
        render_trip_recommendations(recommender)
    elif active_page == "Gender Classification Demo":
        render_gender_demo(users, api_statuses)
    elif active_page == "Analytics":
        render_analytics(flights, hotels, recommender)


if __name__ == "__main__":
    main()
