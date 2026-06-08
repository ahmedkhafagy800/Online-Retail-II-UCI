import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Online Retail Dashboard", layout="wide", page_icon="🛒")

st.title("🛒 Online Retail II — Sales & RFM Dashboard")

# ─── Load & Cache Data ───────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1kD1pRtLEdP0ayeF3CSRGlxPmbClciw88"

@st.cache_data
def load_data():
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    df = pd.read_csv(url)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")

    # ── Description nulls: fill from most-frequent per StockCode ──
    most_freq = (
        df[df["Description"].notnull()]
        .groupby("StockCode")["Description"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"Description": "freq_Description"})
    )
    df = df.merge(most_freq, on="StockCode", how="left")
    df["Description"] = df["Description"].fillna(df["freq_Description"])
    df = df.drop(columns=["freq_Description"])
    df = df[df["Description"].notnull()]

    # ── Remove returns & bad prices ──
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # ── Feature engineering ──
    df["TotalSales"] = df["Quantity"] * df["Price"]
    df["Month"]      = df["InvoiceDate"].dt.month

    return df


@st.cache_data
def build_rfm(df):
    reference_date = df["InvoiceDate"].max()
    rfm = df.groupby("Customer ID").agg(
        Recency   =("InvoiceDate",  lambda x: (reference_date - x.max()).days),
        Frequency =("Invoice",      "nunique"),
        Monetary  =("TotalSales",   "sum"),
    ).reset_index()

    rfm["R_Score"] = pd.qcut(rfm["Recency"],                          q=5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),   q=5, labels=[1,2,3,4,5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"],                         q=5, labels=[1,2,3,4,5])
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    def segment(row):
        r, f, m = int(row["R_Score"]), int(row["F_Score"]), int(row["M_Score"])
        if r >= 4 and f >= 4 and m >= 4:  return "🏆 Champion"
        elif r >= 3 and f >= 3:           return "💛 Loyal"
        elif r >= 4 and f <= 2:           return "🌱 New Customer"
        elif r <= 2 and f >= 3:           return "⚠️ At Risk"
        elif r <= 2 and f <= 2:           return "💤 Lost"
        else:                             return "🔶 Potential"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm


# ─── Load Data ───────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading data from Google Drive..."):
    df  = load_data()
    rfm = build_rfm(df)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Sales Analysis", "👥 RFM Analysis", "🔍 Customer Details"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",    f"£{df['TotalSales'].sum():,.0f}")
    c2.metric("Total Orders",     f"{df['Invoice'].nunique():,}")
    c3.metric("Total Customers",  f"{df['Customer ID'].nunique():,}")
    c4.metric("Total Products",   f"{df['StockCode'].nunique():,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        top5_countries = (
            df.groupby("Country")["TotalSales"]
            .sum().sort_values(ascending=False).head().reset_index()
        )
        fig = px.bar(top5_countries, x="Country", y="TotalSales",
                     title="Top 5 Countries by Revenue", text="TotalSales",
                     color="TotalSales", color_continuous_scale="Blues")
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        fig = px.pie(seg_counts, names="Segment", values="count",
                     title="Customer Segments Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SALES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Sales Analysis")

    # Monthly Sales
    monthly = df.groupby("Month")["TotalSales"].sum().reset_index()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly["MonthName"] = monthly["Month"].apply(lambda x: month_names[x-1])

    fig = px.line(monthly, x="MonthName", y="TotalSales",
                  title="Total Sales per Month (All Years Combined)",
                  markers=True, labels={"MonthName": "Month"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        top5_products = (
            df.groupby("StockCode")["TotalSales"]
            .sum().sort_values(ascending=False).head().reset_index()
        )
        fig = px.bar(top5_products, x="StockCode", y="TotalSales",
                     title="Top 5 Products by Revenue", text="TotalSales",
                     color="TotalSales", color_continuous_scale="Greens")
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top10_countries = (
            df.groupby("Country")["TotalSales"]
            .sum().sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(top10_countries, x="Country", y="TotalSales",
                     title="Top 10 Countries by Revenue", text="TotalSales",
                     color="TotalSales", color_continuous_scale="Oranges")
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RFM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("👥 RFM Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")
    c2.metric("Avg Frequency",      f"{rfm['Frequency'].mean():.1f}")
    c3.metric("Avg Monetary (£)",   f"£{rfm['Monetary'].mean():,.0f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        fig = px.bar(seg_counts, x="Segment", y="count",
                     title="Customer Segments Distribution",
                     color="Segment",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(rfm, x="Recency", y="Monetary",
                         color="Segment", size="Frequency",
                         title="RFM Scatter: Recency vs Monetary",
                         hover_data=["Customer ID"],
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    # RFM Table
    st.subheader("RFM Table")
    st.dataframe(
        rfm[["Customer ID","Recency","Frequency","Monetary","RFM_Score","Segment"]]
        .sort_values("Monetary", ascending=False),
        use_container_width=True, height=400
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Customer Details")

    # Filter by segment
    segments = ["All"] + sorted(rfm["Segment"].unique().tolist())
    selected_seg = st.selectbox("Filter by Segment", segments)

    if selected_seg == "All":
        filtered_rfm = rfm
    else:
        filtered_rfm = rfm[rfm["Segment"] == selected_seg]

    st.write(f"**{len(filtered_rfm):,} customers** in this segment")

    # Search by Customer ID
    search = st.text_input("Search by Customer ID")
    if search:
        filtered_rfm = filtered_rfm[
            filtered_rfm["Customer ID"].astype(str).str.contains(search)
        ]

    st.dataframe(
        filtered_rfm[["Customer ID","Recency","Frequency","Monetary","RFM_Score","Segment"]]
        .sort_values("Monetary", ascending=False),
        use_container_width=True, height=500
    )
