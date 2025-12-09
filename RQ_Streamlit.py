import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime

st.title("Wikipedia Political Interest Analysis (2023–2024)")
st.markdown("---")

st.header("Introduction")
st.markdown("**Research Question:** How do Wikipedia pageviews of political topics vary across countries during the 2023–2024 period, particularly around electoral seasons? Are certain countries more likely to engage with political content than others, and do we see spikes in attention that correspond to major political events?")
st.markdown("**Expectations:** I expect to find increases in political pageviews as a country nears their elections")
st.markdown("**Hypotheses:**")
st.markdown("- Countries undergoing national elections in 2023–2024 will show a significant increase in political article views during election months compared to non-election months.")
st.markdown("- Non-political articles will dominate overall traffic, but political articles will show sharper temporal fluctuations.")
data = {
    "Country": ["United States", "United Kingdom", "Australia", "Canada", "India"],
    "Election Type": [
        "General Election + Primaries",
        "General Election",
        "State/Territory Elections",
        "By-elections (Federal election due 2025)",
        "General Election (Lok Sabha)"
    ],
    "Date Range": [
        "Primaries Jan–Jun 2024; General Nov 5, 2024",
        "Campaign May–Jun 2024; Election Jul 4, 2024",
        "NSW Mar 25, 2023; NT Aug 24, 2024; QLD Oct 26, 2024",
        "Various by-elections in 2023–2024",
        "Apr 19 – Jun 1, 2024 (7 phases); Results Jun 4, 2024"
    ]
}

df_elections = pd.DataFrame(data)

st.header("Major Election Date Ranges (2023–2024)")
st.table(df_elections)

st.markdown("---")

st.header("Data Summary")
st.markdown("The dataset consists of the **top 10,000 most-viewed Wikipedia articles** across five countries (United States, United Kingdom, Canada, Australia, and India) during the years **2023–2024**. Each CSV file contains daily pageviews for articles, along with metadata such as article ID, QID (Wikidata identifier), and project language.")
st.markdown("**Time Interval:** February 2023 – December 2024")
st.markdown("**Wiki:** English Wikipedia (`en.wikipedia`)")
st.markdown("**Countries:** United States, United Kingdom, Canada, Australia, India")

st.markdown("**CSV Files:** 10,000 unique articles per country (top 10,000 viewed) with columns: `date`, `country`, `country_code`, `project`, `page_id`, `article`, `qid`, `views`, `description`, `label`")

st.markdown("---")

st.header("New Features")
st.markdown("**API Calls:** Using the QID columns from the DPDP files, I queiried Wiidata to retrieve unique article descriptions which helped give context to each article (describing a person, event, place, or concept)")
st.markdown("**Classification Feature:** I trained a **Naive Bayes classifier** on political and non-political articles to categorize each article. This produced a new column in the CSV files (label)")

st.markdown("---")


# --- Load Data ---
usa = pd.read_csv("my_data/top_united_states.csv")
uk = pd.read_csv("my_data/top_uk.csv")
canada = pd.read_csv("my_data/top_canada.csv")
australia = pd.read_csv("my_data/top_australia.csv")
india = pd.read_csv("my_data/top_india.csv")

datasets = {
    "USA": usa,
    "UK": uk,
    "Canada": canada,
    "Australia": australia,
    "India": india
}

df = pd.concat(datasets.values(), ignore_index=True)

# Ensure date column is datetime
df['date'] = pd.to_datetime(df['date'])

# --- Dataset Preview ---
st.header("Dataset Overview")
country = st.selectbox("Select a country to preview its dataset:", datasets.keys())
st.write("#### Sample Rows")
st.dataframe(datasets[country].head())
st.write("#### Summary Statistics")
st.write(datasets[country].describe())

# --- Bar Chart: Political vs Non-Political ---
st.header("Political vs Non-Political Articles by Country")

counts = df.groupby(['country', 'label']).size().reset_index(name='count')

fig_bar = px.bar(
    counts,
    x="country",
    y="count",
    color="label",
    barmode="group",
    title="Political vs Non-Political Articles per Country"
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Line Chart: Political Articles Over Time ---
st.header("Political Article Trends Over Time")

# Date range slider
min_date, max_date = df['date'].min().date(), df['date'].max().date()
selected_years = st.slider(
    "Select date range of Wikipedia pageviews to display:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    step=timedelta(days=30)
)

min_selected, max_selected = selected_years
mask = (df['date'].dt.date >= min_selected) & (df['date'].dt.date <= max_selected)
filtered_df = df[mask]

# Aggregate monthly counts of political articles
monthly = (
    filtered_df[filtered_df['label'] == "political"]
    .groupby([pd.Grouper(key='date', freq='M'), 'country'])
    .size()
    .reset_index(name='political_count')
)

selected_country = st.multiselect(
    "Countries to display:",
    options=monthly['country'].unique(),
    default=monthly['country'].unique()
)

monthly = monthly[monthly['country'].isin(selected_country)]

fig_line = px.line(
    monthly,
    x="date",
    y="political_count",
    color="country",
    title="Total Political Articles Over Time"
)

fig_line.update_layout(
    legend_title_text="Country",
    margin=dict(t=80, b=40),
    height=500
)

st.plotly_chart(fig_line, use_container_width=True)

st.markdown("### Monthly Sample Table")
st.dataframe(monthly.head(20))