
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
print(os.listdir())

# --- Dashboard Theme ---
THEME = {
    "background": "#0E1117",   # dark background
    "primary": "#008080",      # teal
    "negative": "#FF8C00"      # orange
}


# --- Force standard column names (FIX FOR KEYERRORS) ---
COLUMN_MAP = {
    'Sentiment_Polarity': 'Sentiment',
    'sentiment_polarity': 'Sentiment',
    'sentiment': 'Sentiment',
    'label': 'Sentiment',

    'Text_Content': 'Text',
    'text_content': 'Text',
    'review': 'Text',
    'comment': 'Text',

    'Bank_Name': 'Bank',
    'bank_name': 'Bank',
    'source_bank': 'Bank',
    'Source_Bank': 'Bank',

    'Platform_Name': 'Platform',
    'platform_name': 'Platform',
    'platform': 'Platform'
}

# --- App Config ---
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #004c4c, #008080);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">
            Nigerian Digital Banking Sentiment Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)




COLOR_THEME = {
    'positive': '#27ae60',   # green
    'neutral': '#95a5a6',    # white
    'negative': '#c0392b'    # red
}


def load_data():
    df = pd.read_csv("master_df_labeled.csv")
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

df = load_data()

for col in list(df.columns):
    if col in COLUMN_MAP:
        df = df.rename(columns={col: COLUMN_MAP[col]})



# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Bank filter
bank_options = sorted(df['Bank'].unique())
selected_banks = st.sidebar.multiselect(
    "Select Bank(s)",
    bank_options,
    default=bank_options[:3]
)

# Platform filter ✅
platform_options = sorted(df['Platform'].unique())
selected_platforms = st.sidebar.multiselect(
    "Select Platform(s)",
    platform_options,
    default=platform_options
)

# Date filter
min_date = df['Date'].min()
max_date = df['Date'].max()

start_date, end_date = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY/MM/DD"
)

# --- Optimized Filtering ---
@st.cache_data(show_spinner=False)
def filter_data(data, banks, platforms, start, end):
    return data[
        (data['Bank'].isin(banks)) &
        (data['Platform'].isin(platforms)) &
        (data['Date'] >= start) &
        (data['Date'] <= end)
    ].copy()

filtered_df = filter_data(df, selected_banks, selected_platforms, start_date, end_date)

if filtered_df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# --- Aggregations (Optimized) ---
total_count = len(filtered_df)

sentiment_counts = (
    filtered_df['Sentiment']
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'Sentiment', 'Sentiment': 'Count'})
)

positive_count = filtered_df[filtered_df['Sentiment'] == 'positive'].shape[0]
negative_count = filtered_df[filtered_df['Sentiment'] == 'negative'].shape[0]

net_score = (positive_count - negative_count) / total_count
net_score_percent = f"{net_score * 100:.2f}%"

filtered_df['Week'] = filtered_df['Date'].dt.to_period('W').astype(str)

trend_df = (
    filtered_df
    .groupby(['Week', 'Sentiment'])
    .size()
    .reset_index(name='Count')
)

trend_df['Date'] = pd.to_datetime(trend_df['Week'].str.split('/').str[0])

# ✅ Bank comparison aggregation
bank_sentiment = (
    filtered_df
    .groupby(['Bank', 'Sentiment'])
    .size()
    .reset_index(name='Count')
)


# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div style="padding:15px; background-color:"#0E0E11"; border-radius:10px;
                    border-left:6px solid #7EC87FF";">
                <p style="color:white; margin:0;">Total Feedback</p>
                <h2 style="color:"#7EC87FF"; margin:0;">{total_count:,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="padding:15px; background-color:"#0E0E11"; border-radius:10px;
                    border-left:6px solid "#7EC87FF";">
                <p style="color:white; margin:0;">Positive (%)</p>
                <h2 style="color:"#7EC87FF"; margin:0;">{(positive_count/total_count)*100:.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="padding:15px; background-color:"#0E0E11"; border-radius:10px;
                    border-left:6px solid "#7EC87FF";">
                <p style="color:white; margin:0;">Negative (%)</p>
                <h2 style="color: "#c0392b"; margin:0;">{(negative_count/total_count)*100:.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
       st.markdown(
        f"""
        <div style="padding:15px; background-color:"#0E0E11"; border-radius:10px;
                    border-left:6px solid "#7EC87FF";">
                <p style="color:white; margin:0;">Net Sentiment</p>
                <h2 style="color:"#0057B8; margin:0;">{net_score_percent}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
 

st.markdown("---")

# --- Safe Sentiment Aggregation ---
if 'Sentiment' not in filtered_df.columns:
    st.error("Sentiment column is missing. Columns available:")
    st.write(filtered_df.columns.tolist())
    st.stop()

sentiment_counts = (
    filtered_df[['Sentiment']]
    .value_counts()
    .reset_index()
)

sentiment_counts.columns = ['Sentiment', 'Count']


# --- Charts Row 1 ---
col_pie, col_line = st.columns([1,2])

with col_pie:
    st.subheader("Sentiment Distribution")

    fig_pie = px.pie(
    names=sentiment_counts['Sentiment'],
    values=sentiment_counts['Count'],
    color_discrete_map =  {
    'positive': '#27ae60',   # green
    'neutral': '#95a5a6',    # white
    'negative': '#c0392b'    # red
}
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with col_line:
    st.subheader("Weekly Sentiment Trend")

    fig_line = px.line(
        trend_df,
        x='Date',
        y='Count',
        color='Sentiment',
        color_discrete_map= {
    'positive': '#27ae60',   # green
    'neutral': '#95a5a6',    # white
    'negative': '#c0392b'    # red
}
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ✅ --- Bank Comparison Chart ---
st.markdown("---")
st.subheader("Bank Comparison: Sentiment Volume")

fig_bank = px.bar(
    bank_sentiment,
    x='Bank',
    y='Count',
    color='Sentiment',
    barmode='group',
    color_discrete_map= {
    'positive': '#27ae60',   # green
    'neutral': '#95a5a6',    # white
    'negative': '#c0392b'    # red
}
)
st.plotly_chart(fig_bank, use_container_width=True)

# --- Theme colors (define once) ---
THEME = {
    "background": "#0E1117",
    "text": "#FFFFFF"
}

st.subheader("Word Clouds by Sentiment")

col_wc1, col_wc2 = st.columns([1, 1])


# POSITIVE WORD CLOUD
with col_wc1:
    st.markdown("### Positive Feedback")

    pos_text = " ".join(
        filtered_df[filtered_df['Sentiment'] == 'positive']['Text'].astype(str)
    )

    if len(pos_text) > 10:
        wc_pos = WordCloud(
            width=1200,
            height=600,
            background_color=THEME["background"],
            colormap="viridis"
        ).generate(pos_text)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.imshow(wc_pos, interpolation="bilinear")
        ax1.axis("off")
        st.pyplot(fig1)
    else:
        st.info("Not enough positive text.")


# NEGATIVE WORD CLOUD
with col_wc2:
    st.markdown("### Negative Feedback")

    neg_text = " ".join(
        filtered_df[filtered_df['Sentiment'] == 'negative']['Text'].astype(str)
    )

    if len(neg_text) > 10:
        wc_neg = WordCloud(
            width=1200,
            height=600,
            background_color=THEME["background"],
            colormap="autumn"
        ).generate(neg_text)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.imshow(wc_neg, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)
    else:
        st.info("Not enough negative text.")


# --- Recent Feedback Table ---
col_table, = st.columns(1)

col_table.subheader("Recent Feedback")

pos = filtered_df[
    (filtered_df['Sentiment'] == 'positive') &
    (filtered_df['Platform'] != 'Mixed')
].sort_values('Date', ascending=False)[['Date','Bank','Platform','Text']].head(10)

neg = filtered_df[
    (filtered_df['Sentiment'] == 'negative') &
    (filtered_df['Platform'] != 'Mixed')
].sort_values('Date', ascending=False)[['Date','Bank','Platform','Text']].head(10)

t1, t2 = st.tabs(["Top Positive", "Top Negative"])

with t1:
    st.dataframe(pos, use_container_width=True)

with t2:
    st.dataframe(neg, use_container_width=True)

