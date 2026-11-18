# Buyer Segmentation Dashboard - Parcl Co. Limited
# Run segmentation_analysis.py first (it creates outputs/segmented_clients.csv), then:
#     streamlit run buyer_streamlit_app.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Buyer Segmentation Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("outputs/segmented_clients.csv")


df = load_data()
segments = list(df.sort_values("cluster")["segment"].unique())   # fixed order for every chart

st.title("Buyer Segmentation & Investment Profiling Dashboard")
st.caption("Parcl Co. Limited - Real Estate Market Intelligence")

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
country_pick = st.sidebar.multiselect("Country (empty = all)", sorted(df["country"].unique()))
region_pick = st.sidebar.multiselect("Region (empty = all)", sorted(df["region"].unique()))
purpose_pick = st.sidebar.multiselect("Acquisition purpose", sorted(df["acquisition_purpose"].unique()),
                                      default=sorted(df["acquisition_purpose"].unique()))
type_pick = st.sidebar.multiselect("Client type", sorted(df["client_type"].unique()),
                                   default=sorted(df["client_type"].unique()))

data = df[df["acquisition_purpose"].isin(purpose_pick) & df["client_type"].isin(type_pick)]
if country_pick:
    data = data[data["country"].isin(country_pick)]
if region_pick:
    data = data[data["region"].isin(region_pick)]

st.sidebar.write(f"**{len(data)}** clients match the filters")
if data.empty:
    st.warning("No clients match these filters. Please change the filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Segmentation Overview", "Investor Behavior",
                                  "Geographic Buyer Analysis", "Segment Insights"])

# ---------------- tab 1: overview ----------------
with tab1:
    a, b, c, d = st.columns(4)
    a.metric("Clients", len(data))
    b.metric("Segments", data["segment"].nunique())
    c.metric("Avg satisfaction", round(data["satisfaction_score"].mean(), 2))
    d.metric("Avg units bought", round(data["units_bought"].mean(), 2))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        sns.countplot(y="segment", data=data, order=segments, ax=ax)
        ax.set_title("Clients in each segment")
        ax.set_ylabel("")
        st.pyplot(fig)
    with right:
        share = data["segment"].value_counts(normalize=True).mul(100).round(1)
        st.subheader("Share of each segment (%)")
        st.dataframe(share.rename("Share (%)").to_frame())

# ---------------- tab 2: investor behaviour ----------------
with tab2:
    left, right = st.columns(2)
    with left:
        mix = pd.crosstab(data["segment"], data["acquisition_purpose"], normalize="index").mul(100)
        fig2, ax2 = plt.subplots()
        mix.reindex(segments).dropna().plot(kind="barh", stacked=True, ax=ax2)
        ax2.set_title("Acquisition purpose in each segment (%)")
        ax2.set_ylabel("")
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        st.pyplot(fig2)
    with right:
        fig3, ax3 = plt.subplots()
        sns.barplot(y="segment", x="loan_num", data=data, order=segments, errorbar=None, ax=ax3)
        ax3.set_title("Loan rate in each segment")
        ax3.set_ylabel("")
        st.pyplot(fig3)

    left, right = st.columns(2)
    with left:
        fig4, ax4 = plt.subplots()
        sns.barplot(y="segment", x="units_bought", data=data, order=segments, errorbar=None, ax=ax4)
        ax4.set_title("Average units bought")
        ax4.set_ylabel("")
        st.pyplot(fig4)
    with right:
        fig5, ax5 = plt.subplots()
        sns.barplot(y="segment", x="total_spend", data=data, order=segments, errorbar=None, ax=ax5)
        ax5.set_title("Average total spend (USD)")
        ax5.set_ylabel("")
        st.pyplot(fig5)

    channel = pd.crosstab(data["segment"], data["referral_channel"], normalize="index").mul(100).round(1)
    st.subheader("How each segment found Parcl (referral channel, %)")
    st.dataframe(channel.reindex(segments).dropna())

# ---------------- tab 3: geography ----------------
with tab3:
    pick = st.selectbox("Show clients of segment", ["All segments"] + segments)
    geo = data if pick == "All segments" else data[data["segment"] == pick]

    st.subheader("Clients by country")
    by_country = geo.groupby("country_iso3").size().reset_index(name="clients")
    st.plotly_chart(px.choropleth(by_country, locations="country_iso3", color="clients",
                                  color_continuous_scale="Blues"), width="stretch")

    st.subheader("Clients by US state")
    by_state = geo.dropna(subset=["state_code"]).groupby("state_code").size().reset_index(name="clients")
    if by_state.empty:
        st.info("No US clients under the current filters.")
    else:
        st.plotly_chart(px.choropleth(by_state, locations="state_code", locationmode="USA-states",
                                      scope="usa", color="clients", color_continuous_scale="Blues"),
                        width="stretch")

    st.subheader("Segments in the top 12 regions")
    top_regions = data["region"].value_counts().head(12).index
    reg = pd.crosstab(data[data["region"].isin(top_regions)]["region"], data["segment"])
    fig6, ax6 = plt.subplots(figsize=(9, 4))
    reg.loc[top_regions].plot(kind="bar", stacked=True, ax=ax6)
    ax6.set_xlabel("")
    ax6.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig6)

    st.subheader("Segments by country (number of clients)")
    fig7, ax7 = plt.subplots(figsize=(9, 4))
    present = [s for s in segments if s in data["segment"].unique()]
    sns.heatmap(pd.crosstab(data["country"], data["segment"])[present], annot=True, fmt="d", cmap="Blues", ax=ax7)
    ax7.set_ylabel("")
    ax7.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig7)

# ---------------- tab 4: segment insights ----------------
with tab4:
    st.subheader("Descriptive statistics for each segment")
    stats = data.groupby("segment").agg(
        clients=("client_id", "count"),
        avg_age=("age", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_units=("units_bought", "mean"),
        avg_unit_price=("avg_unit_price", "mean"),
        avg_total_spend=("total_spend", "mean"),
        loan_rate=("loan_num", "mean"),
        office_share=("office_share", "mean"),
    ).round(2)
    st.dataframe(stats.reindex([s for s in segments if s in stats.index]))

    st.subheader("Look inside one segment")
    chosen = st.selectbox("Segment", [s for s in segments if s in stats.index])
    one = data[data["segment"] == chosen]
    a, b, c = st.columns(3)
    a.metric("Clients", len(one))
    b.metric("Average age", round(one["age"].mean(), 1))
    c.metric("Average total spend", f"${one['total_spend'].mean():,.0f}")
    for col in ["client_type", "acquisition_purpose", "country", "referral_channel"]:
        st.write(f"**{col.replace('_', ' ').title()}**")
        st.dataframe(one[col].value_counts(normalize=True).mul(100).round(1).rename("Share (%)").head(6).to_frame())

    st.download_button("Download segmented clients (CSV)", data.to_csv(index=False), "segmented_clients.csv")
