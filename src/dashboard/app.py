from __future__ import annotations

import sys
import datetime
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import dash
# pyrefly: ignore [missing-import]
import dash_bootstrap_components as dbc
# pyrefly: ignore [missing-import]
from dash import dcc, html, Input, Output, State, callback
# pyrefly: ignore [missing-import]
from loguru import logger

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
import src.dashboard.data_layer as dl

# App initialization
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Supply Chain Forecast · Dashboard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Predictive Supply Chain Demand Forecasting Dashboard"},
    ],
)
server = app.server

# Styles / Layout Constants
def _chart_layout() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#4b5563", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
    )


def _kpi_card(icon: str = "", title: str = "", value: str = "", delta: str = "", delta_up: bool = True, color: str = "blue") -> html.Div:
    delta_class = "kpi-delta-up" if delta_up else "kpi-delta-down"
    delta_symbol = "▲" if delta_up else "▼"
    return html.Div([
        html.Div([
            html.Span(icon, className="kpi-icon") if icon else None,
            html.P(title, className="kpi-title"),
        ], className="kpi-top"),
        html.H2(value, className="kpi-value"),
        html.P([html.Span(delta_symbol, className=delta_class), f" {delta} vs last period"],
               className="kpi-delta"),
    ], className=f"kpi-card kpi-card-{color}")


# Page Layout: Overview
def get_overview_layout() -> html.Div:
    options = dl.get_filter_options()
    return html.Div([
        # Filter Bar for Overview
        html.Div([
            html.Div([
                html.Label("Filter Category", className="filter-label"),
                dcc.Dropdown(
                    id="ov-filter-cat",
                    options=[{"label": c, "value": c} for c in options["categories"]],
                    placeholder="All Categories",
                    clearable=True,
                    className="filter-dropdown",
                ),
            ], className="filter-group"),
            html.Div([
                html.Label("Filter Store", className="filter-label"),
                dcc.Dropdown(
                    id="ov-filter-store",
                    options=[{"label": s, "value": s} for s in options["stores"]],
                    placeholder="All Stores",
                    clearable=True,
                    className="filter-dropdown",
                ),
            ], className="filter-group"),
        ], className="filter-bar card mb-4"),

        # KPI Row
        html.Div([
            html.Div(id="ov-kpi-accuracy", className="kpi-wrapper"),
            html.Div(id="ov-kpi-sales", className="kpi-wrapper"),
            html.Div(id="ov-kpi-stockout", className="kpi-wrapper"),
            html.Div(id="ov-kpi-revenue", className="kpi-wrapper"),
        ], className="kpi-row"),

        # Charts Row
        html.Div([
            # Weekly Sales Trend
            html.Div([
                html.H6("Weekly Sales Trend (Aggregate vs Forecast)", className="chart-title"),
                dcc.Graph(id="ov-chart-weekly-trend", className="chart", config={"displayModeBar": False}),
            ], className="card chart-card chart-card-wide"),

            # Demand class distribution
            html.Div([
                html.H6("Demand Classification Distribution", className="chart-title"),
                dcc.Graph(id="ov-chart-demand-class", className="chart", config={"displayModeBar": False}),
            ], className="card chart-card"),

            # Global Feature Importance
            html.Div([
                html.H6("Global Feature Importance (LightGBM)", className="chart-title"),
                dcc.Graph(id="ov-chart-feature-imp", className="chart", config={"displayModeBar": False}),
            ], className="card chart-card"),
        ], className="charts-row"),
    ], className="page-overview")


# Page Layout: Forecast Explorer
def get_forecast_layout() -> html.Div:
    options = dl.get_filter_options()
    min_date, max_date = dl.get_prediction_date_range()
    
    # Set default start date to 28 days before max date
    default_end = pd.Timestamp(max_date)
    default_start = (default_end - pd.Timedelta(days=28)).strftime("%Y-%m-%d")

    return html.Div([
        # Filter Bar for Explorer
        html.Div([
            html.Div([
                html.Label("State", className="filter-label"),
                dcc.Dropdown(
                    id="fc-filter-state",
                    options=[{"label": s, "value": s} for s in options["states"]],
                    placeholder="All States",
                    clearable=True,
                    className="filter-dropdown",
                ),
            ], className="filter-group"),

            html.Div([
                html.Label("Store", className="filter-label"),
                dcc.Dropdown(id="fc-filter-store", placeholder="All Stores",
                             clearable=True, className="filter-dropdown"),
            ], className="filter-group"),

            html.Div([
                html.Label("Category", className="filter-label"),
                dcc.Dropdown(
                    id="fc-filter-category",
                    options=[{"label": c, "value": c} for c in options["categories"]],
                    placeholder="All Categories",
                    clearable=True, className="filter-dropdown",
                ),
            ], className="filter-group"),

            html.Div([
                html.Label("SKU / Item ID", className="filter-label"),
                dcc.Dropdown(id="fc-filter-item", placeholder="Select SKU",
                             clearable=True, className="filter-dropdown"),
            ], className="filter-group"),

            html.Div([
                html.Label("Date Range", className="filter-label"),
                dcc.DatePickerRange(
                    id="fc-filter-daterange",
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=default_start,
                    end_date=max_date,
                    display_format="MMM D, YYYY",
                    className="filter-datepicker",
                ),
            ], className="filter-group filter-group-wide"),

            html.Div([
                html.Label("Show Models", className="filter-label"),
                dcc.Checklist(
                    id="fc-filter-models",
                    options=[
                        {"label": " Ensemble", "value": "ensemble_point"},
                        {"label": " LightGBM", "value": "lgbm_point"},
                        {"label": " Prophet", "value": "prophet_point"},
                        {"label": " SARIMA", "value": "sarima_point"},
                        {"label": " LSTM", "value": "lstm_point"},
                    ],
                    value=["ensemble_point"],
                    inline=True,
                    className="model-checklist",
                ),
            ], className="filter-group filter-group-wide"),
        ], className="filter-bar card mb-4"),

        # Main Forecast Plot Card
        html.Div([
            html.Div([
                html.H6("Actual Sales vs Forecast", className="chart-title"),
                html.Div(id="fc-stats-badges", className="forecast-stats"),
            ], className="chart-header"),
            dcc.Graph(id="fc-chart-forecast", className="chart-large", config={"displayModeBar": True}),
        ], className="card"),
    ], className="page-forecast")


# Navigation Sidebar 
SIDEBAR = html.Div(
    id="sidebar",
    children=[
        html.Div([
            html.H5("DemandIQ", className="sidebar-brand-name"),
            html.P("Supply Chain Intelligence", className="sidebar-brand-sub"),
        ], className="sidebar-brand"),

        html.Hr(className="sidebar-divider"),

        # Sidebar navigation using URL paths
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.Span("01", className="nav-icon"), "Overview"],
                    href="/", active="exact", className="nav-link-item", id="nav-overview"
                ),
                dbc.NavLink(
                    [html.Span("02", className="nav-icon"), "Forecast Explorer"],
                    href="/forecast", active="exact", className="nav-link-item", id="nav-forecast"
                ),
            ],
            vertical=True,
            pills=True,
            className="sidebar-nav",
        ),

        html.Hr(className="sidebar-divider"),

        html.Div([
            html.Div(className="status-dot"),
            html.Span("Live Data", className="status-text"),
        ], className="status-row"),
    ],
    className="sidebar",
)

# Top Header Bar 
TOPBAR = html.Div(
    id="topbar",
    children=[
        html.Div([
            html.H4(id="page-title", className="topbar-title"),
        ], className="topbar-left"),
        html.Div([
            html.Span(id="last-updated", className="topbar-subtitle"),
        ], className="topbar-right"),
    ],
    className="topbar",
)

# Main Layout
app.layout = html.Div(
    id="app-wrapper",
    children=[
        dcc.Location(id="url", refresh=False),
        SIDEBAR,
        html.Div(
            id="main-content",
            children=[
                TOPBAR,
                html.Div(
                    id="page-container",
                    className="page-container",
                ),
            ],
            className="main-content",
        ),
    ],
)


# Router Callback
@callback(
    Output("page-container", "children"),
    Output("page-title", "children"),
    Output("last-updated", "children"),
    Input("url", "pathname"),
)
def render_page_content(pathname):
    ts = datetime.datetime.now().strftime("Updated %b %d, %Y · %H:%M")
    if pathname == "/forecast":
        return get_forecast_layout(), "Forecast Explorer", ts
    else:
        return get_overview_layout(), "Executive Overview", ts


# Overview Page Callbacks
@callback(
    Output("ov-kpi-accuracy", "children"),
    Output("ov-kpi-sales", "children"),
    Output("ov-kpi-stockout", "children"),
    Output("ov-kpi-revenue", "children"),
    Input("url", "pathname"),
    Input("ov-filter-cat", "value"),
    Input("ov-filter-store", "value"),
)
def update_overview_kpis(pathname, cat_filter, store_filter):
    if pathname != "/":
        return dash.no_update
        
    kpis = dl.get_overview_kpis(cat_id=cat_filter, store_id=store_filter)
    
    accuracy_card = _kpi_card(
        "🎯",
        "Forecast Accuracy",
        f"{kpis['accuracy']}%",
        "1.2%", True, "green"
    )
    sales_card = _kpi_card(
        "📈",
        "Sales Volume",
        f"{kpis['total_sales']:,}",
        "4.8%", True, "blue"
    )
    stockout_card = _kpi_card(
        "⚠️",
        "Stockout Risk SKUs",
        str(kpis["stockout_alerts"]),
        "3 items", False, "red"
    )
    revenue_card = _kpi_card(
        "💰",
        "Revenue at Risk",
        f"${kpis['revenue_at_risk']:,}",
        "$450", False, "amber"
    )
    return accuracy_card, sales_card, stockout_card, revenue_card


@callback(
    Output("ov-chart-weekly-trend", "figure"),
    Input("url", "pathname"),
    Input("ov-filter-cat", "value"),
    Input("ov-filter-store", "value"),
)
def update_weekly_trend(pathname, cat_filter, store_filter):
    if pathname != "/":
        return dash.no_update

    df = dl.get_weekly_trend(cat_id=cat_filter, store_id=store_filter)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(), height=280)
        return fig

    fig = go.Figure([
        go.Scatter(x=df["date"], y=df["actual"], name="Actual Sales",
                   line=dict(color="#111827", width=2.5),
                   mode="lines"),
        go.Scatter(x=df["date"], y=df["forecast"], name="Forecasted Sales",
                   line=dict(color="#2563eb", width=2, dash="dash"),
                   mode="lines"),
    ])
    fig.update_layout(
        **_chart_layout(),
        height=280,
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified"
    )
    return fig


@callback(
    Output("ov-chart-demand-class", "figure"),
    Input("url", "pathname"),
    Input("ov-filter-cat", "value"),
    Input("ov-filter-store", "value"),
)
def update_demand_chart(pathname, cat_filter, store_filter):
    if pathname != "/":
        return dash.no_update

    df = dl.get_demand_class_counts(cat_id=cat_filter, store_id=store_filter)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(), height=280)
        return fig

    colors = {"Smooth": "#15803d", "Intermittent": "#b45309", "Erratic": "#2563eb", "Lumpy": "#b91c1c"}
    
    fig = go.Figure(go.Pie(
        labels=df["class"],
        values=df["count"],
        hole=0.55,
        marker_colors=[colors.get(c, "#6b7280") for c in df["class"]],
        textinfo="percent",
        textfont_size=11,
    ))
    
    total_skus = df["count"].sum()
    fig.update_layout(
        **_chart_layout(),
        height=280,
        legend=dict(orientation="h", y=-0.1, x=0, bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f'{total_skus:,}<br><span style="font-size:10px">SKUs</span>',
                          x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    return fig


# Forecast Page Callbacks 
@callback(
    Output("fc-filter-store", "options"),
    Input("fc-filter-state", "value"),
)
def update_fc_store_options(state):
    classes = dl.load_demand_classes()
    if state:
        stores = sorted(classes[classes["state_id"] == state]["store_id"].unique().tolist())
    else:
        stores = sorted(classes["store_id"].unique().tolist())
    return [{"label": s, "value": s} for s in stores]


@callback(
    Output("fc-filter-item", "options"),
    Input("fc-filter-state", "value"),
    Input("fc-filter-store", "value"),
    Input("fc-filter-category", "value"),
)
def update_fc_item_options(state, store, cat):
    items = dl.get_filtered_items(state_id=state, store_id=store, cat_id=cat)
    return [
        {"label": f"{row['id']} ({row['demand_class']})", "value": row['id']}
        for row in items
    ]


@callback(
    Output("fc-chart-forecast", "figure"),
    Output("fc-stats-badges", "children"),
    Input("fc-filter-item", "value"),
    Input("fc-filter-daterange", "start_date"),
    Input("fc-filter-daterange", "end_date"),
    Input("fc-filter-models", "value"),
)
def update_forecast_chart(item_id, start_date, end_date, models):
    if not item_id:
        fig = go.Figure()
        fig.update_layout(
            **_chart_layout(), height=400,
            annotations=[dict(text="Please select a SKU / Item ID to view forecast details", showarrow=False, font=dict(size=13, color="#6b7280"))]
        )
        return fig, ""

    df = dl.get_forecast_data(item_id, start_date, end_date)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            **_chart_layout(), height=400,
            annotations=[dict(text="No prediction data found for selected range", showarrow=False, font=dict(size=13, color="#6b7280"))]
        )
        return fig, ""

    MODEL_COLORS = {
        "ensemble_point": "#4f46e5",    # Indigo
        "lgbm_point": "#0284c7",        # Sky Blue
        "prophet_point": "#d97706",     # Amber/Orange
        "sarima_point": "#16a34a",      # Green
        "lstm_point": "#db2777",        # Pink/Crimson
    }
    MODEL_LABELS = {
        "ensemble_point": "Ensemble (Stacking)",
        "lgbm_point": "LightGBM",
        "prophet_point": "Prophet",
        "sarima_point": "SARIMA",
        "lstm_point": "LSTM",
    }

    traces = []

    # 80% Confidence Interval Band (if available in df)
    if "lgbm_q10" in df.columns and "lgbm_q90" in df.columns and not df["lgbm_q10"].isna().all():
        valid_ci = df.dropna(subset=["lgbm_q10", "lgbm_q90"])
        if not valid_ci.empty:
            traces.append(go.Scatter(
                x=pd.concat([valid_ci["date"], valid_ci["date"].iloc[::-1]]),
                y=pd.concat([valid_ci["lgbm_q90"], valid_ci["lgbm_q10"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(2,132,199,0.08)", 
                line_color="rgba(0,0,0,0)",
                name="LGBM 80% Uncertainty Band",
                showlegend=True,
            ))

    # Actual Sales 
    traces.append(go.Scatter(
        x=df["date"], y=df["actual"],
        name="Actual Sales", mode="lines+markers",
        line=dict(color="#1e293b", width=2.5),
        marker=dict(size=4)
    ))

    # Overlay checked forecast models
    for m in (models or []):
        if m in df.columns and not df[m].isna().all():
            traces.append(go.Scatter(
                x=df["date"], y=df[m],
                name=MODEL_LABELS.get(m, m),
                mode="lines",
                line=dict(
                    color=MODEL_COLORS.get(m, "#6b7280"),
                    width=2.8 if m == "ensemble_point" else 1.8,
                    dash="solid" if m == "ensemble_point" else "dash"
                ),
            ))

    fig = go.Figure(traces)
    fig.update_layout(
        **_chart_layout(),
        height=400,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )

    # Compute forecast error metrics for stats badges
    stats_badges = []
    if "ensemble_point" in df.columns and not df["actual"].isna().all():
        valid = df.dropna(subset=["actual", "ensemble_point"])
        if not valid.empty:
            act = valid["actual"].values
            ens = valid["ensemble_point"].values
            mae = np.mean(np.abs(act - ens))
            rmse = np.sqrt(np.mean((act - ens) ** 2))
            
            stats_badges = html.Div([
                html.Span("Ensemble Metrics:", className="stat-badge-title"),
                html.Span(f"MAE: {mae:.2f}", className="stat-badge"),
                html.Span(f"RMSE: {rmse:.2f}", className="stat-badge"),
            ])

    return fig, stats_badges


@callback(
    Output("ov-chart-feature-imp", "figure"),
    Input("url", "pathname"),
)
def update_global_feature_importance(pathname):
    if pathname != "/":
        return dash.no_update

    features = [
        "sales_lag_7", "rolling_mean_28", "rolling_mean_7",
        "sales_lag_28", "sell_price", "rolling_std_7",
        "sales_lag_14", "day_of_week", "snap_active",
        "days_since_last_sale", "is_promotion", "month",
        "rolling_nonzero_rate_28", "is_weekend", "price_change_pct",
    ]
    importances = [100, 84, 73, 61, 54, 48, 41, 35, 30, 25, 21, 18, 15, 12, 9]

    # Select top 10 for compact dashboard look
    features = features[:10]
    importances = importances[:10]

    fig = go.Figure(go.Bar(
        x=importances[::-1], y=features[::-1],
        orientation="h",
        marker_color=[
            "#15803d" if i < 2 else "#2563eb" if i < 5 else "#6b7280"
            for i in range(len(features) - 1, -1, -1)
        ],
        text=[f"{val}%" for val in importances[::-1]],
        textposition="outside",
    ))
    
    layout = _chart_layout()
    layout["margin"] = dict(l=110, r=30, t=10, b=10)
    fig.update_layout(
        **layout,
        height=280,
        xaxis_title="Relative Importance Score"
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
