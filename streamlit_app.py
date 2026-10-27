
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="VANTARA | Customer Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ROOT = Path(__file__).resolve().parent

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    [data-testid="stSidebar"] {
        display: none;
    }

    .stApp {
        background: #071321;
        color: #e6edf5;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.5rem 2rem 3rem 2rem;
    }

    h1, h2, h3 {
        color: #e6edf5 !important;
    }

    p, label, span {
        color: #c3d0df;
    }

    [data-testid="stMetric"] {
        background: #102235;
        border: 1px solid #1d374e;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfd4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f6fb !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #091827;
        border: 1px solid #1b344b;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #20394f;
        border-radius: 8px;
    }

    div[data-baseweb="select"] {
        background: #102235;
    }

    hr {
        border-color: #20394f;
    }

    .subtitle {
        color: #91a8bd;
        margin-top: -12px;
        margin-bottom: 22px;
    }

    .section-note {
        color: #91a8bd;
        font-size: 13px;
    }

    .status {
        color: #32d6b0;
        font-size: 13px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# 2. LOAD PROJECT DATA
def find_file(*paths):
    for relative_path in paths:
        path = ROOT / relative_path
        if path.exists():
            return path
    return None


def load_table(*paths):
    path = find_file(*paths)

    if path is None:
        return pd.DataFrame()

    try:
        if path.suffix.lower() in [".pkl", ".pickle"]:
            data = pd.read_pickle(path)
        else:
            data = pd.read_csv(path)

        data.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in data.columns
        ]

        return data

    except Exception as error:
        st.warning(f"Could not load {path.name}: {error}")
        return pd.DataFrame()


customers = load_table(
    "data/processed/customer_features_train.csv",
    "data/processed/customer_features.csv",
    "customer_features_train.csv",
    "customer_features.csv"
)

transactions = load_table(
    "data/processed/cleaned_transaction.pkl",
    "data/processed/cleaned_transaction.csv",
    "cleaned_transaction.pkl",
    "cleaned_transaction.csv"
)

segments = load_table(
    "data/processed/customer_segments.csv",
    "customer_segments.csv"
)

model_results = load_table(
    "reports/classical_model_comparison.csv",
    "data/processed/classical_model_comparison.csv",
    "classical_model_comparison.csv"
)

# 3. PREPARE CUSTOMER DATA
def get_column(data, possible_names):
    for name in possible_names:
        if name in data.columns:
            return name
    return None


customer_id_col = get_column(
    customers, ["customer_id", "customer id", "customerid"]
)

frequency_col = get_column(
    customers, ["frequency", "purchase_frequency", "total_purchases"]
)

monetary_col = get_column(
    customers, ["monetary", "total_spend", "total_spending",
                "historical_clv", "transaction_amount"]
)

recency_col = get_column(
    customers, ["recency", "recency_days", "days_since_last_purchase"]
)

basket_col = get_column(
    customers, ["basket_size", "avg_basket_size"]
)

average_spend_col = get_column(
    customers, ["average_spend", "avg_spend", "average_order_value"]
)

date_col = get_column(
    transactions, ["invoicedate", "invoice_date", "date"]
)

invoice_col = get_column(
    transactions, ["invoice", "invoiceno", "invoice_no", "invoice_number"]
)

quantity_col = get_column(
    transactions, ["quantity"]
)

price_col = get_column(
    transactions, ["price", "unit_price"]
)

transaction_customer_col = get_column(
    transactions, ["customer_id", "customerid"]
)

if not customers.empty:
    if customer_id_col is None:
        st.error("Customer features file has no recognizable customer ID column.")
        st.stop()

    customers = customers.dropna(subset=[customer_id_col]).copy()

    if monetary_col is None:
        customers["monetary"] = 0.0
        monetary_col = "monetary"

    if frequency_col is None:
        customers["frequency"] = 0
        frequency_col = "frequency"

    if recency_col is None:
        customers["recency"] = 0
        recency_col = "recency"

    for column in [monetary_col, frequency_col, recency_col]:
        customers[column] = pd.to_numeric(
            customers[column], errors="coerce"
        ).fillna(0)

    if average_spend_col is None:
        customers["average_spend"] = (
            customers[monetary_col]
            / customers[frequency_col].replace(0, 1)
        )
        average_spend_col = "average_spend"

    if basket_col is None:
        customers["basket_size"] = 0
        basket_col = "basket_size"

    customers[average_spend_col] = pd.to_numeric(
        customers[average_spend_col], errors="coerce"
    ).fillna(0)

    customers[basket_col] = pd.to_numeric(
        customers[basket_col], errors="coerce"
    ).fillna(0)

# 4. PREPARE TRANSACTIONS AND MONTHLY DATA
if not transactions.empty and date_col:
    transactions[date_col] = pd.to_datetime(
        transactions[date_col], errors="coerce"
    )

    transactions = transactions.dropna(subset=[date_col]).copy()

    if "transaction_amount" in transactions.columns:
        transactions["transaction_amount"] = pd.to_numeric(
            transactions["transaction_amount"], errors="coerce"
        ).fillna(0)

    elif quantity_col and price_col:
        transactions["transaction_amount"] = (
            pd.to_numeric(transactions[quantity_col], errors="coerce").fillna(0)
            * pd.to_numeric(transactions[price_col], errors="coerce").fillna(0)
        )

    else:
        transactions["transaction_amount"] = 0.0

    # Exclude cancellation invoices from the purchase count.
    if invoice_col:
        transactions["_is_cancellation"] = (
            transactions[invoice_col]
            .astype(str)
            .str.upper()
            .str.startswith("C")
        )
    else:
        transactions["_is_cancellation"] = False

    if "is_cancellation" in transactions.columns:
        transactions["_is_cancellation"] |= (
            transactions["is_cancellation"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    purchases = transactions[
        ~transactions["_is_cancellation"]
    ].copy()

    purchases["_month"] = purchases[date_col].dt.to_period("M").astype(str)

    monthly_sales = (
        purchases.groupby("_month")["transaction_amount"]
        .sum()
        .reset_index(name="sales")
    )

    if invoice_col:
        monthly_purchases = (
            purchases.groupby("_month")[invoice_col]
            .nunique()
            .reset_index(name="purchases")
        )
    else:
        monthly_purchases = (
            purchases.groupby("_month")
            .size()
            .reset_index(name="purchases")
        )

    if transaction_customer_col:
        monthly_customers = (
            purchases.groupby("_month")[transaction_customer_col]
            .nunique()
            .reset_index(name="active_customers")
        )
    else:
        monthly_customers = pd.DataFrame(
            columns=["_month", "active_customers"]
        )

    monthly = monthly_sales.merge(
        monthly_purchases, on="_month", how="outer"
    )

    monthly = monthly.merge(
        monthly_customers, on="_month", how="left"
    )

    monthly = monthly.fillna(0).sort_values("_month")

else:
    purchases = pd.DataFrame()
    monthly = pd.DataFrame(
        columns=["_month", "sales", "purchases", "active_customers"]
    )

# 5. PREPARE CUSTOMER SEGMENTS
if not customers.empty:
    segment_col = get_column(
        customers,
        ["segment", "customer_segment", "segment_name", "cluster"]
    )

    if segment_col is None and not segments.empty:
        segment_id_col = get_column(
            segments, ["customer_id", "customerid"]
        )
        segment_label_col = get_column(
            segments, ["segment", "customer_segment",
                       "segment_name", "cluster"]
        )

        if segment_id_col and segment_label_col:
            segment_lookup = segments[
                [segment_id_col, segment_label_col]
            ].drop_duplicates(segment_id_col)

            customers = customers.merge(
                segment_lookup,
                left_on=customer_id_col,
                right_on=segment_id_col,
                how="left",
                suffixes=("", "_segment")
            )

            segment_col = segment_label_col

    if segment_col is None:
        def classify_recency(days):
            if days <= 90:
                return "Loyal"
            elif days <= 180:
                return "At Risk"
            elif days <= 365:
                return "Potential Churn"
            return "Lost"

        customers["display_segment"] = customers[recency_col].apply(
            classify_recency
        )
        segment_col = "display_segment"


# 6. HEADER
header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown("#   VANTARA")
    st.markdown(
        '<p class="subtitle">Customer Analytics Dashboard<br>'
        'Better insights. Stronger customer relationships.</p>',
        unsafe_allow_html=True
    )

with header_right:
    if not customers.empty:
        st.markdown(
            '<p class="status">● Customer data loaded</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status">● Check data files</p>',
            unsafe_allow_html=True
        )

if customers.empty:
    st.error(
        "Customer data could not be loaded. Check that "
        "customer_features_train.csv exists in data/processed/."
    )
    st.stop()

# 7. OVERVIEW
st.markdown("## ◉ Overview")
st.caption("Key metrics and business performance at a glance.")

total_customers = len(customers)

if not purchases.empty:
    total_sales = purchases["transaction_amount"].sum()
    if invoice_col:
        total_purchases = purchases[invoice_col].nunique()
    else:
        total_purchases = len(purchases)
else:
    total_sales = customers[monetary_col].sum()
    total_purchases = customers[frequency_col].sum()

average_spend = customers[average_spend_col].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Total Sales", f"₹ {total_sales:,.0f}")
c3.metric("Average Spend", f"₹ {average_spend:,.2f}")
c4.metric("Monthly Purchases", f"{total_purchases:,.0f}")

st.divider()

# 8. CUSTOMER DATA
st.markdown("## ♙ Customer Data")
st.caption("Customer information and purchase behaviour.")

display_columns = [
    (customer_id_col, "Customer ID"),
    (frequency_col, "Purchase Frequency"),
    (monetary_col, "Total Spend"),
    (recency_col, "Recency (Days)"),
    (basket_col, "Basket Size"),
    (average_spend_col, "Avg. Spend"),
]

table_columns = {}
for original, label in display_columns:
    if original and original in customers.columns:
        table_columns[original] = label

customer_table = customers[list(table_columns)].rename(
    columns=table_columns
)

if "Total Spend" in customer_table:
    customer_table["Total Spend"] = customer_table[
        "Total Spend"
    ].round(2)

if "Avg. Spend" in customer_table:
    customer_table["Avg. Spend"] = customer_table[
        "Avg. Spend"
    ].round(2)

st.dataframe(
    customer_table,
    use_container_width=True,
    height=240,
    hide_index=True
)

st.caption(f"Showing {len(customer_table):,} of {total_customers:,} customers.")

st.divider()

# 9. TOP 10 LOYAL VIP CUSTOMERS
st.markdown("## 👑 Top 10 Loyal VIP Customers")
st.caption(
    "Highest-spending customers ranked by historical spend. "
    "This is a spending leaderboard, not a churn prediction."
)

vip = customers.sort_values(
    monetary_col, ascending=False
).head(10).copy()

vip_table = pd.DataFrame({
    "Rank": range(1, len(vip) + 1),
    "Customer ID": vip[customer_id_col].astype(str).values,
    "Total Spend": vip[monetary_col].round(2).values,
    "Purchase Frequency": vip[frequency_col].values,
    "Recency (Days)": vip[recency_col].values
})

st.dataframe(
    vip_table,
    use_container_width=True,
    height=260,
    hide_index=True
)

st.divider()

# 10. CUSTOMER SEGMENT INSIGHTS
st.markdown("## ◔ Customer Segment Insights")
st.caption("Distribution of customers across different segments.")

# Assign the four display segments consistently using recency.
def classify_customer(days):
    if days <= 90:
        return "Loyal"
    elif days <= 180:
        return "At Risk"
    elif days <= 365:
        return "Potential Churn"
    return "Lost"


segment_data = customers.copy()
segment_data["_display_segment"] = segment_data[recency_col].apply(
    classify_customer
)

segment_order = ["Loyal", "At Risk", "Potential Churn", "Lost"]

segment_colors = {
    "Loyal": "#10b99a",
    "At Risk": "#318cf5",
    "Potential Churn": "#f6b33e",
    "Lost": "#8956d8",
}

segment_behaviour = {
    "Loyal": "High spending, frequent",
    "At Risk": "Declining activity",
    "Potential Churn": "Infrequent purchases",
    "Lost": "No recent activity",
}

# Calculate segment counts and business metrics.
segment_rows = []

for name in segment_order:
    group = segment_data[
        segment_data["_display_segment"] == name
    ]

    segment_rows.append({
        "Segment": name,
        "Customers": len(group),
        "Percentage": (
            len(group) / len(segment_data) * 100
            if len(segment_data) else 0
        ),
        "Avg. Spend": group[average_spend_col].mean()
        if not group.empty else 0,
        "Avg. Recency": group[recency_col].mean()
        if not group.empty else 0,
        "Key Behaviour": segment_behaviour[name],
    })

segment_summary = pd.DataFrame(segment_rows)

# Main layout: chart and legend on the left, overview table on the right.
left_panel, right_panel = st.columns([1.05, 1.55], gap="large")

with left_panel:
    pie_col, legend_col = st.columns([1.0, 1.35], gap="small")

    with pie_col:
        fig_segments = px.pie(
            segment_summary,
            names="Segment",
            values="Customers",
            hole=0,
            color="Segment",
            category_orders={"Segment": segment_order},
            color_discrete_map=segment_colors,
        )

        fig_segments.update_traces(
            texttemplate="%{percent:.0%}",
            textposition="inside",
            textfont=dict(size=13, color="white"),
            sort=False,
            direction="clockwise",
            rotation=90,
            hovertemplate=(
                "%{label}<br>"
                "%{value:,} customers<br>"
                "%{percent}<extra></extra>"
            ),
        )

        fig_segments.update_layout(
            template="plotly_dark",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=5, b=5, l=0, r=0),
            height=285,
        )

        st.plotly_chart(
            fig_segments,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with legend_col:
        st.markdown("<div style='height:25px'></div>",
                    unsafe_allow_html=True)

        for _, row in segment_summary.iterrows():
            color = segment_colors[row["Segment"]]

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                    padding:13px 0;
                    border-bottom:1px solid #1b3044;
                    font-size:13px;
                ">
                    <span style="
                        width:12px;
                        height:12px;
                        min-width:12px;
                        border-radius:50%;
                        background:{color};
                        display:inline-block;
                    "></span>
                    <span style="flex:1;color:#d5e1ee;">
                        {row['Segment']}
                    </span>
                    <span style="
                        color:#d5e1ee;
                        white-space:nowrap;
                    ">
                        {row['Customers']:,}
                        ({row['Percentage']:.0f}%)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right_panel:
    st.markdown("### Segment Overview")

    # Build a compact table matching the reference design.
    table_html = """
    <div style="
        border:1px solid #1d374e;
        border-radius:9px;
        overflow:hidden;
        background:#0b1b2b;
        width:100%;
    ">
    <table style="
        width:100%;
        border-collapse:collapse;
        font-size:12px;
        color:#d5e1ee;
    ">
    <thead>
    <tr style="background:#12263a;color:#a9bfd4;">
        <th style="padding:12px 8px;text-align:left;">Segment</th>
        <th style="padding:12px 8px;text-align:left;">Avg. Spend</th>
        <th style="padding:12px 8px;text-align:left;">Avg. Recency</th>
        <th style="padding:12px 8px;text-align:left;">Key Behaviour</th>
    </tr>
    </thead>
    <tbody>
    """

    for _, row in segment_summary.iterrows():
        color = segment_colors[row["Segment"]]

        table_html += f"""
        <tr style="border-top:1px solid #1b3044;">
            <td style="padding:13px 8px;white-space:nowrap;">
                <span style="color:{color};">●</span>
                {row['Segment']}
            </td>
            <td style="padding:13px 8px;white-space:nowrap;">
                ₹ {row['Avg. Spend']:,.2f}
            </td>
            <td style="padding:13px 8px;white-space:nowrap;">
                {row['Avg. Recency']:.0f} days
            </td>
            <td style="padding:13px 8px;">
                {row['Key Behaviour']}
            </td>
        </tr>
        """

    table_html += "</tbody></table></div>"

    st.html(table_html)

st.caption(
    "Segments are assigned using recency: Loyal ≤90 days, "
    "At Risk 91–180 days, Potential Churn 181–365 days, "
    "and Lost >365 days. These are rule-based groups, "
    "not churn-model predictions."
)

st.divider()

# 11. MODEL PERFORMANCE

st.markdown("## ▥ Model Performance")
st.caption("Comparison of machine-learning models on evaluation metrics.")

if not model_results.empty:
    model_name_col = get_column(
        model_results, ["model", "model_name", "algorithm"]
    )

    if model_name_col:
        model_results = model_results.rename(
            columns={model_name_col: "Model"}
        )

        metric_aliases = {
            "accuracy": ["accuracy", "acc"],
            "precision": ["precision"],
            "recall": ["recall"],
            "f1_score": ["f1_score", "f1-score", "f1"],
            "roc_auc": ["roc_auc", "roc-auc", "auc"]
        }

        metric_rename = {}

        for standard, aliases in metric_aliases.items():
            found = get_column(model_results, aliases)
            if found:
                metric_rename[found] = standard

        model_results = model_results.rename(
            columns=metric_rename
        )

        metric_columns = [
            col for col in [
                "accuracy", "precision", "recall",
                "f1_score", "roc_auc"
            ]
            if col in model_results.columns
        ]

        for col in metric_columns:
            model_results[col] = pd.to_numeric(
                model_results[col], errors="coerce"
            )

        if metric_columns:
            st.dataframe(
                model_results[["Model"] + metric_columns].round(3),
                use_container_width=True,
                hide_index=True
            )

            chart_columns = st.columns(
                min(3, len(metric_columns))
            )

            for index, metric in enumerate(metric_columns[:3]):
                with chart_columns[index]:
                    chart_data = model_results[
                        ["Model", metric]
                    ].dropna()

                    fig = px.bar(
                        chart_data,
                        x="Model",
                        y=metric,
                        title=metric.replace("_", " ").title(),
                        color="Model",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        margin=dict(t=50, b=15, l=10, r=10),
                        xaxis_title="",
                        yaxis_title=""
                    )

                    st.plotly_chart(
                        fig, use_container_width=True
                    )
        else:
            st.warning(
                "The model evaluation file was found, but no "
                "recognized evaluation metric columns were detected."
            )
    else:
        st.warning(
            "The model results file needs a recognizable model-name column."
        )
else:
    st.info(
        "Model evaluation file not found. Add your actual model "
        "comparison CSV to reports/ or data/processed/ to display metrics."
    )

st.divider()

# 12. SALES AND CUSTOMER TRENDS
st.markdown("## ↗ Sales & Customer Trends")
st.caption("Monthly sales, purchase activity and average spend.")

if not monthly.empty:
    trend1, trend2 = st.columns(2)

    with trend1:
        fig_sales = px.area(
            monthly,
            x="_month",
            y="sales",
            title="Monthly Sales Revenue",
            markers=True
        )

        fig_sales.update_traces(
            line_color="#10b99a",
            fillcolor="rgba(16,185,154,0.18)"
        )

        fig_sales.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Sales (₹)",
            margin=dict(t=50, b=20, l=10, r=10)
        )

        st.plotly_chart(fig_sales, use_container_width=True)

    with trend2:
        fig_activity = px.line(
            monthly,
            x="_month",
            y="active_customers",
            title="Monthly Active Customers",
            markers=True
        )

        fig_activity.update_traces(
            line_color="#318cf5",
            line_width=3
        )

        fig_activity.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Unique Purchasing Customers",
            margin=dict(t=50, b=20, l=10, r=10)
        )

        st.plotly_chart(fig_activity, use_container_width=True)

    trend3, trend4 = st.columns(2)

    with trend3:
        fig_purchases = px.bar(
            monthly,
            x="_month",
            y="purchases",
            title="Monthly Purchase Count",
            color_discrete_sequence=["#10b99a"]
        )

        fig_purchases.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Purchases",
            margin=dict(t=50, b=20, l=10, r=10)
        )

        st.plotly_chart(fig_purchases, use_container_width=True)

    with trend4:
        monthly_average = monthly.copy()

        monthly_average["average_spend"] = (
            monthly_average["sales"]
            / monthly_average["active_customers"].replace(0, float("nan"))
        )

        fig_average = px.line(
            monthly_average,
            x="_month",
            y="average_spend",
            title="Sales per Active Customer",
            markers=True
        )

        fig_average.update_traces(
            line_color="#a879ed",
            line_width=3
        )

        fig_average.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Sales per Customer (₹)",
            margin=dict(t=50, b=20, l=10, r=10)
        )

        st.plotly_chart(fig_average, use_container_width=True)

    # Selected month details
    st.markdown("### Monthly Purchase Details")

    selected_month = st.selectbox(
        "Select Month",
        options=monthly["_month"].tolist(),
        index=len(monthly) - 1
    )

    selected = monthly[
        monthly["_month"] == selected_month
    ].iloc[0]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Total Sales",
        f"₹ {selected['sales']:,.2f}"
    )

    m2.metric(
        "Purchase Count",
        f"{selected['purchases']:,.0f}"
    )

    m3.metric(
        "Active Customers",
        f"{selected['active_customers']:,.0f}"
    )

else:
    st.info(
        "Monthly trends cannot be displayed because transaction data "
        "or a recognizable invoice date column was not found."
    )

# 13. FOOTER
st.divider()

st.caption(
    "VANTARA | Customer Analytics Dashboard | "
    "Historical data insights for customer relationship management"
)
