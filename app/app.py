from pathlib import Path
from openai import OpenAI

import pandas as pd
import streamlit as st
import plotly.express as px

import time

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = None

start = time.time()

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Hospital Readmissions Analytics",
    layout="wide"
)

st.markdown("""
<style>

/* Keep background white */
.stApp {
    background-color: white;
}

/* Headings */
h1, h2, h3,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    color: #16324F !important;
}

/* KPI cards */
[data-testid="stMetric"] {
    background-color: #EAF2FF;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #BFD3F2;
}

[data-testid="stMetricLabel"] {
    color: #16324F !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #16324F !important;
}

/* Filter boxes */
div[data-baseweb="select"] > div {
    background-color: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
}

/* Filter labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-weight: 700 !important;
    color: #16324F !important;
}

/* Primary button */
div.stButton > button {
    background-color: #2563EB !important;
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)




st.title("AI-Powered Hospital Readmissions Analytics")

st.caption(
    "Interactive analysis of CMS hospital readmission performance "
    "with grounded GenAI-generated executive insights."
)

st.markdown("### Understanding the Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Excess Readmission Ratio (ERR)**  
    Compares a hospital's **predicted readmissions** with the number expected
    for hospitals treating similar patients.

    - **ERR > 1:** Above expected
    - **ERR = 1:** At expected
    - **ERR < 1:** Below expected
    """)

with col2:
    st.markdown("""
    **Predicted Readmission Rate**  
    The hospital's estimated **risk-adjusted readmission rate**, accounting
    for the types of patients treated.
    """)

with col3:
    st.markdown("""
    **Expected Readmission Rate**  
    The readmission rate expected if the hospital's patients were treated
    at an average hospital with similar patients.
    """)

st.info(
    "Example: An ERR of 1.05 means predicted readmissions are approximately "
    "5% above the expected benchmark. ERR measures performance relative to "
    "expected readmissions—it is not the same as the absolute readmission rate."
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "healthcare_readmissions_combined.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()



#client = OpenAI(
#   api_key=st.secrets["OPENAI_API_KEY"]
#)



condition_labels = {
    "READM-30-AMI-HRRP": "Heart Attack (AMI)",
    "READM-30-CABG-HRRP": "CABG Surgery",
    "READM-30-COPD-HRRP": "COPD",
    "READM-30-HF-HRRP": "Heart Failure",
    "READM-30-HIP-KNEE-HRRP": "Hip / Knee Replacement",
    "READM-30-PN-HRRP": "Pneumonia"
}



# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")


# State filter
state_options = sorted(
    df["State"]
    .dropna()
    .unique()
)

selected_state = st.sidebar.selectbox(
    "Select State",
    state_options
)


# Condition filter
condition_options = sorted(
    df["Measure Name"]
    .dropna()
    .unique()
)

selected_condition = st.sidebar.selectbox(
    "Select Condition",
    condition_options,
    format_func=lambda x: condition_labels.get(x, x)
)


# Hospital ownership filter
ownership_options = [
    "All"
] + sorted(
    df["Hospital Ownership"]
    .dropna()
    .unique()
)

selected_ownership = st.sidebar.selectbox(
    "Hospital Ownership",
    ownership_options
)


# Hospital rating filter
rating_options = [
    "All",
    "1",
    "2",
    "3",
    "4",
    "5"
]

selected_rating = st.sidebar.selectbox(
    "Hospital Overall Rating",
    rating_options
)


# ---------------------------------------------------------
# Apply filters
# ---------------------------------------------------------
base_filtered_df = df[
    (df["State"] == selected_state)
    &
    (df["Measure Name"] == selected_condition)
].copy()

if selected_ownership != "All":
    base_filtered_df = base_filtered_df[
        base_filtered_df["Hospital Ownership"]
        == selected_ownership
    ]

# This one is for KPIs, table, AI, and hospital-level chart
filtered_df = base_filtered_df.copy()

if selected_rating != "All":
    filtered_df = filtered_df[
        filtered_df["Hospital overall rating"].astype(str)
        == selected_rating
    ]

# ---------------------------------------------------------
# Handle empty results
# ---------------------------------------------------------
if filtered_df.empty:
    st.warning(
        "No hospitals match the selected combination of filters."
    )

    st.stop()


# ---------------------------------------------------------
# Calculate KPIs
# ---------------------------------------------------------
avg_err = filtered_df[
    "Excess Readmission Ratio"
].mean()

avg_predicted_rate = filtered_df[
    "Predicted Readmission Rate"
].mean()

avg_expected_rate = filtered_df[
    "Expected Readmission Rate"
].mean()

reporting_hospitals = filtered_df[
    "Facility ID"
].nunique()

above_expected_pct = (
    filtered_df[
        "Excess Readmission Ratio"
    ]
    .gt(1)
    .mean()
    * 100
)


# ---------------------------------------------------------
# Selected-filter summary
# ---------------------------------------------------------
selected_condition_label = condition_labels.get(
    selected_condition,
    selected_condition
)

st.subheader(
    f"{selected_state} | {selected_condition_label}"
)

filter_description = []

if selected_ownership != "All":
    filter_description.append(
        f"Ownership: {selected_ownership}"
    )

if selected_rating != "All":
    filter_description.append(
        f"Rating: {selected_rating}-star"
    )

if filter_description:
    st.caption(
        " | ".join(filter_description)
    )


# ---------------------------------------------------------
# KPI section
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label="Average ERR",
    value=f"{avg_err:.3f}"
)

col2.metric(
    label="Avg Predicted Rate",
    value=f"{avg_predicted_rate:.2f}%"
)

col3.metric(
    label="Avg Expected Rate",
    value=f"{avg_expected_rate:.2f}%"
)

col4.metric(
    label="Reporting Hospitals",
    value=reporting_hospitals
)

col5.metric(
    label="Hospitals Above Expected",
    value=f"{above_expected_pct:.1f}%"
)





import altair as alt

# ---------------------------------------------------------
# Hospital ERR comparison chart
# ---------------------------------------------------------
st.subheader("Hospital ERR Comparison")

chart_data = (
    filtered_df[
        [
            "Facility ID",
            "Facility Name",
            "Excess Readmission Ratio"
        ]
    ]
    .copy()
)

# Ensure ERR is numeric
chart_data["Excess Readmission Ratio"] = pd.to_numeric(
    chart_data["Excess Readmission Ratio"],
    errors="coerce"
)

# Clean hospital names
chart_data["Facility Name"] = (
    chart_data["Facility Name"]
    .astype(str)
    .str.strip()
)

# Remove rows without usable ERR
chart_data = chart_data.dropna(
    subset=["Excess Readmission Ratio"]
)

# Replace blank names with Facility ID
chart_data.loc[
    chart_data["Facility Name"].isin(["", "nan", "None"]),
    "Facility Name"
] = (
    "Facility ID "
    + chart_data["Facility ID"].astype(str)
)

# Sort and keep top 10
chart_data = (
    chart_data
    .sort_values(
        "Excess Readmission Ratio",
        ascending=False
    )
    .head(10)
)

# Convert ERR to percentage difference from expected
chart_data["Difference from Expected"] = (
    chart_data["Excess Readmission Ratio"] - 1
) * 100

# Readable status
chart_data["Benchmark Status"] = chart_data[
    "Difference from Expected"
].apply(
    lambda x: "Above expected"
    if x > 0
    else "Below expected"
    if x < 0
    else "At expected"
)

# Text displayed at end of bar
chart_data["Difference Label"] = chart_data[
    "Difference from Expected"
].map(
    lambda x: f"{x:+.1f}%"
)

if chart_data.empty:

    st.info(
        "No hospital-level ERR data are available "
        "for the selected filters."
    )

else:

    fig = px.bar(
        chart_data,
        x="Difference from Expected",
        y="Facility Name",
        orientation="h",
        color="Benchmark Status",
        text="Difference Label",
        hover_data={
            "Facility Name": True,
            "Excess Readmission Ratio": ":.3f",
            "Difference from Expected": ":.1f",
            "Benchmark Status": False
        },
        labels={
            "Facility Name": "",
            "Difference from Expected": "% Difference from Expected"
        },
	color_discrete_map={
        "Above expected": "#C0392B",
        "Below expected": "#2E8B57",
        "At expected": "#6B7280"
    }
    )

    # Expected benchmark
    fig.add_vline(
        x=0,
        line_dash="dash"
    )

    # Keep highest ERR at the top
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=chart_data[
            "Facility Name"
        ].tolist()[::-1]
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=max(
            350,
            len(chart_data) * 45
        ),
        legend_title_text="",
        margin=dict(
            l=20,
            r=60,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "0% represents the expected benchmark (ERR = 1.0). "
        "Positive values indicate predicted readmissions above expected; "
        "negative values indicate predicted readmissions below expected."
    )

# ---------------------------------------------------------
# Average ERR by Hospital Rating
# ---------------------------------------------------------
st.subheader("Average ERR by Hospital Rating")

# Use the base filtered data so all rating groups remain
# available for comparison.
rating_source = base_filtered_df.copy()

# Keep only valid ratings
rating_source = rating_source[
    rating_source["Hospital overall rating"]
    .astype(str)
    .isin(["1", "2", "3", "4", "5"])
].copy()

# Calculate one average ERR per hospital first
hospital_rating_data = (
    rating_source
    .groupby(
        [
            "Facility ID",
            "Hospital overall rating"
        ],
        as_index=False
    )
    .agg(
        Hospital_Avg_ERR=(
            "Excess Readmission Ratio",
            "mean"
        )
    )
)

# Then calculate average ERR across hospitals in each rating group
rating_chart_data = (
    hospital_rating_data
    .groupby(
        "Hospital overall rating",
        as_index=False
    )
    .agg(
        Avg_ERR=(
            "Hospital_Avg_ERR",
            "mean"
        ),
        Hospitals=(
            "Facility ID",
            "nunique"
        )
    )
)

# Convert ERR into % difference from expected
rating_chart_data["Difference from Expected"] = (
    rating_chart_data["Avg_ERR"] - 1
) * 100

# Readable rating labels
rating_chart_data["Rating"] = (
    rating_chart_data["Hospital overall rating"]
    .astype(str)
    + "-Star"
)

# Labels shown above bars
rating_chart_data["Difference Label"] = (
    rating_chart_data["Difference from Expected"]
    .map(lambda x: f"{x:+.1f}%")
)

# Classify each rating group relative to expected
rating_chart_data["Benchmark Status"] = (
    rating_chart_data["Difference from Expected"]
    .apply(
        lambda x: "Above expected"
        if x > 0
        else "Below expected"
        if x < 0
        else "At expected"
    )
)

if rating_chart_data.empty:

    st.info(
        "No hospital rating data are available "
        "for the selected filters."
    )

else:

    rating_fig = px.bar(
        rating_chart_data,
        x="Rating",
        y="Difference from Expected",
        color="Benchmark Status",
        text="Difference Label",
        hover_data={
            "Rating": False,
            "Avg_ERR": ":.3f",
            "Hospitals": True,
            "Difference from Expected": ":.1f",
            "Benchmark Status": False
        },
        labels={
            "Rating": "Hospital Overall Rating",
            "Difference from Expected": "% Difference from Expected",
            "Avg_ERR": "Average ERR"
        },
        category_orders={
            "Rating": [
                "1-Star",
                "2-Star",
                "3-Star",
                "4-Star",
                "5-Star"
            ],
            "Benchmark Status": [
                "Above expected",
                "Below expected",
                "At expected"
            ]
        },
	color_discrete_map={
        "Above expected": "#C0392B",
        "Below expected": "#2E8B57",
        "At expected": "#6B7280"
    }

    )

    # ERR = 1 corresponds to 0%
    rating_fig.add_hline(
        y=0,
        line_dash="dash"
    )

    rating_fig.update_traces(
        textposition="outside"
    )

    rating_fig.update_layout(
        height=400,
        legend_title_text="",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        rating_fig,
        use_container_width=True
    )

    st.caption(
        "0% represents the expected benchmark (ERR = 1.0). "
        "This chart responds to the selected state, condition, "
        "and ownership filters while retaining all rating groups "
        "for comparison. Associations do not establish causality."
    )


# ---------------------------------------------------------
# Average ERR by Hospital Ownership
# ---------------------------------------------------------
st.subheader("Average ERR by Hospital Ownership")

# Start from state + condition filtered data.
# Deliberately ignore the ownership filter itself so
# all ownership groups remain available for comparison.
ownership_source = df[
    (df["State"] == selected_state)
    &
    (df["Measure Name"] == selected_condition)
].copy()

# Apply the selected hospital rating filter, if any
if selected_rating != "All":
    ownership_source = ownership_source[
        ownership_source["Hospital overall rating"]
        .astype(str)
        == selected_rating
    ]

# Keep rows with usable ownership and ERR values
ownership_source = ownership_source.dropna(
    subset=[
        "Hospital Ownership",
        "Excess Readmission Ratio"
    ]
)

# ---------------------------------------------------------
# Step 1: Calculate one average ERR per hospital
# ---------------------------------------------------------
hospital_ownership_data = (
    ownership_source
    .groupby(
        [
            "Facility ID",
            "Hospital Ownership"
        ],
        as_index=False
    )
    .agg(
        Hospital_Avg_ERR=(
            "Excess Readmission Ratio",
            "mean"
        )
    )
)

# ---------------------------------------------------------
# Step 2: Compare ownership groups
# ---------------------------------------------------------
ownership_chart_data = (
    hospital_ownership_data
    .groupby(
        "Hospital Ownership",
        as_index=False
    )
    .agg(
        Avg_ERR=(
            "Hospital_Avg_ERR",
            "mean"
        ),
        Hospitals=(
            "Facility ID",
            "nunique"
        )
    )
)

# ---------------------------------------------------------
# Step 3: Convert ERR to % difference from expected
# ---------------------------------------------------------
ownership_chart_data["Difference from Expected"] = (
    ownership_chart_data["Avg_ERR"] - 1
) * 100

# Text shown at the end of each bar
ownership_chart_data["Difference Label"] = (
    ownership_chart_data["Difference from Expected"]
    .map(lambda x: f"{x:+.1f}%")
)

# Classify above / below benchmark
ownership_chart_data["Benchmark Status"] = (
    ownership_chart_data["Difference from Expected"]
    .apply(
        lambda x: "Above expected"
        if x > 0
        else "Below expected"
        if x < 0
        else "At expected"
    )
)

# Flag very small ownership groups
ownership_chart_data["Sample Size"] = (
    ownership_chart_data["Hospitals"]
    .apply(
        lambda x: "Small sample (<3 hospitals)"
        if x < 3
        else "3+ hospitals"
    )
)

# Sort chart from lowest to highest difference
ownership_chart_data = (
    ownership_chart_data
    .sort_values(
        "Difference from Expected",
        ascending=True
    )
)

# ---------------------------------------------------------
# Step 4: Display chart
# ---------------------------------------------------------
if ownership_chart_data.empty:

    st.info(
        "No ownership comparison is available "
        "for the selected filters."
    )

else:

    ownership_fig = px.bar(
        ownership_chart_data,
        x="Difference from Expected",
        y="Hospital Ownership",
        orientation="h",
        color="Benchmark Status",
        text="Difference Label",
        hover_data={
            "Hospital Ownership": False,
            "Avg_ERR": ":.3f",
            "Hospitals": True,
            "Difference from Expected": ":.1f",
            "Benchmark Status": False,
            "Sample Size": True
        },
        labels={
            "Hospital Ownership": "",
            "Difference from Expected": "% Difference from Expected",
            "Avg_ERR": "Average ERR"
        },
        category_orders={
            "Benchmark Status": [
                "Above expected",
                "Below expected",
                "At expected"
            ]
        },
	color_discrete_map={
        "Above expected": "#C0392B",
        "Below expected": "#2E8B57",
        "At expected": "#6B7280"
    }

    )

    # 0% corresponds to ERR = 1.0
    ownership_fig.add_vline(
        x=0,
        line_dash="dash"
    )

    ownership_fig.update_traces(
        textposition="outside"
    )

    ownership_fig.update_layout(
        height=max(
            350,
            len(ownership_chart_data) * 45
        ),
        legend_title_text="",
        margin=dict(
            l=20,
            r=70,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        ownership_fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Small-sample warning
    # -----------------------------------------------------
    if (
        ownership_chart_data["Hospitals"] < 3
    ).any():

        st.warning(
            "One or more ownership groups contain fewer than "
            "3 hospitals. These results are descriptive and "
            "should be interpreted cautiously."
        )

    st.caption(
        "0% represents the expected benchmark (ERR = 1.0). "
        "The chart responds to the selected state, condition, "
        "and rating filters while retaining ownership groups "
        "for comparison. Hospital counts are shown on hover; "
        "very small groups should be interpreted cautiously. "
        "Associations do not establish causality."
    )

# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------
st.subheader("Readmission Performance")

if avg_err > 1:
    relative_difference = (
        avg_err - 1
    ) * 100

    st.write(
        f"The average Excess Readmission Ratio is "
        f"{avg_err:.3f}, meaning predicted readmissions "
        f"are approximately {relative_difference:.1f}% "
        f"above the expected benchmark."
    )

elif avg_err < 1:
    relative_difference = (
        1 - avg_err
    ) * 100

    st.write(
        f"The average Excess Readmission Ratio is "
        f"{avg_err:.3f}, meaning predicted readmissions "
        f"are approximately {relative_difference:.1f}% "
        f"below the expected benchmark."
    )

else:
    st.write(
        "The average Excess Readmission Ratio is "
        "approximately equal to the expected benchmark."
    )


# ---------------------------------------------------------
# Hospital-level table
# ---------------------------------------------------------
st.subheader("Hospital-Level Results")

display_columns = [
    "Facility Name",
    "Hospital Ownership",
    "Hospital overall rating",
    "Excess Readmission Ratio",
    "Predicted Readmission Rate",
    "Expected Readmission Rate"
]

hospital_table = (
    filtered_df[
        display_columns
    ]
    .sort_values(
        "Excess Readmission Ratio",
        ascending=False
    )
)

# Create display copy
hospital_table_display = hospital_table.copy()

# Rename columns for cleaner display
hospital_table_display = hospital_table_display.rename(columns={
    "Hospital overall rating": "Hospital Rating",
    "Excess Readmission Ratio": "ERR",
    "Predicted Readmission Rate": "Predicted Rate",
    "Expected Readmission Rate": "Expected Rate"
})

# Format numeric columns
hospital_table_display["ERR"] = (
    hospital_table_display["ERR"]
    .map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
)

hospital_table_display["Predicted Rate"] = (
    hospital_table_display["Predicted Rate"]
    .map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
)

hospital_table_display["Expected Rate"] = (
    hospital_table_display["Expected Rate"]
    .map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
)

# Convert dataframe to HTML
table_html = hospital_table_display.to_html(
    index=False,
    classes="hospital-table",
    border=0
)

# CSS + HTML table
styled_html = f"""
<style>

.hospital-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
    font-size: 14px;
}}

.hospital-table th {{
    background-color: #16324F;
    color: white;
    font-weight: 600;
    text-align: left;
    padding: 12px;
    border: 1px solid #DCE3EA;
}}

.hospital-table td {{
    padding: 10px 12px;
    border: 1px solid #DCE3EA;
    color: #1F2937;
}}

/* Emphasize ERR column */
.hospital-table td:nth-child(4) {{
    background-color: #FFF4E5;
    font-weight: 700;
    color: #8A4B08;
}}

.hospital-table tr:nth-child(even) {{
    background-color: #F8FAFC;
}}

.hospital-table tr:nth-child(odd) {{
    background-color: white;
}}

.hospital-table tr:hover {{
    background-color: #EEF4FF;
}}

</style>

{table_html}
"""

st.html(styled_html)

# ---------------------------------------------------------
# AI-generated insights
# ---------------------------------------------------------
st.subheader("AI-Generated Insights")

if st.button("✨ Generate AI Insights", type="primary"):

    if not api_key:
        st.warning(
            "AI insights are temporarily unavailable. "
            "The dashboard and analytical results remain available."
        )

    else:

        try:
            client = OpenAI(api_key=api_key)
    	   verified_metrics = {
                   "state": selected_state,
                   "condition": selected_condition,
                   "hospital_ownership": selected_ownership,
                   "hospital_rating": selected_rating,
                   "average_ERR": round(avg_err, 6),
                   "average_predicted_readmission_rate": round(
                    avg_predicted_rate, 6
                     ),
                   "average_expected_readmission_rate": round(
                   avg_expected_rate, 6
                     ),
                   "reporting_hospitals": int(reporting_hospitals),
                   "percent_hospitals_above_expected": round(
                   above_expected_pct, 1
                    )
    }

    	   prompt = f"""
     You are assisting a healthcare operations analyst.

     Use ONLY the verified metrics provided below.

     Do not invent causes, clinical explanations, patient characteristics,
     hospital attributes, or additional statistics.

     Provide:

     1. Key observed findings
     2. Analytical interpretation
     3. What cannot be concluded
     4. Recommended follow-up analysis
     5. Executive summary in no more than 3 sentences

    Important:
    - ERR > 1 means predicted readmissions are above the expected benchmark.
    - ERR < 1 means predicted readmissions are below expected.
    - Distinguish absolute readmission rates from performance relative to expected.
    - Treat ownership and rating findings as associations, not causal effects.
    - Do not describe a hospital, state, ownership type, or rating group as poor performing.
    - Mention sample-size limitations where relevant.
    - Apply conclusions only to the hospitals represented in the filtered data.

         Verified metrics:

         {verified_metrics}
          """

           with st.spinner("Generating AI insights..."):

           response = client.responses.create(
              model="gpt-5.6-luna",
              input=prompt
            )

          ai_output = response.output_text

         st.markdown(ai_output)

        except Exception:
            st.error(
                "AI insights could not be generated at the moment. "
                "Please try again later."
            )

# ---------------------------------------------------------
# Interpretation note
# ---------------------------------------------------------
st.info(
    "ERR measures readmission performance relative to an "
    "expected benchmark. Differences should be interpreted "
    "as associations and do not establish causality or "
    "overall hospital quality."
)




with st.expander("About this project and methodology"):

    st.markdown("""
    ### Project objective

    This project explores hospital readmission performance using
    publicly available CMS Hospital Readmissions Reduction Program
    (HRRP) data enriched with CMS Hospital General Information.

    The application allows users to explore readmission performance
    by state, condition, hospital ownership, and overall hospital rating.

    ### Excess Readmission Ratio

    The Excess Readmission Ratio (ERR) compares a hospital's
    predicted readmission rate with its expected readmission rate.

    - **ERR > 1:** predicted readmissions are above expected.
    - **ERR < 1:** predicted readmissions are below expected.
    - **ERR = 1:** predicted readmissions are approximately equal
      to the expected benchmark.

    For the charts, ERR is translated into percentage difference
    from expected:

    - ERR 1.05 = approximately 5% above expected
    - ERR 0.95 = approximately 5% below expected

    ### Analytical approach

    The analysis includes:

    - State- and condition-level readmission comparisons
    - Hospital-level ERR benchmarking
    - Comparison of ERR across hospital rating groups
    - Comparison of ERR across hospital ownership groups
    - Sample-size checks to avoid overinterpreting very small groups

    For rating and ownership comparisons, hospital-level ERR is first
    calculated so that each hospital contributes equally to the group average.

    ### Interactive filtering

    Dashboard metrics and hospital-level results respond to the selected
    state, condition, ownership, and rating filters.

    Comparison charts intentionally retain the dimension being compared.
    For example, the hospital-rating chart keeps all rating groups visible
    while responding to the selected state, condition, and ownership filters.

    ### AI workflow

    Python calculates all numerical metrics used by the application.

    When **Generate AI Insights** is selected, only the verified metrics
    calculated from the filtered data are passed to the language model.

    The model is instructed to:

    - Use only the supplied metrics
    - Avoid inventing clinical explanations or unsupported statistics
    - Separate observed findings from interpretation
    - Distinguish absolute readmission rates from ERR
    - Treat ownership and rating relationships as associations rather than
      causal effects
    - Acknowledge sample-size and data limitations

    ### Important limitations

    - Results apply only to hospitals with reported CMS data.
    - Missing or suppressed CMS values may reduce the number of hospitals
      available for some analyses.
    - Associations between hospital characteristics and ERR do not establish
      causality.
    - Hospital ownership, rating, geography, patient mix, service mix, and
      other unmeasured factors may be related to the observed patterns.
    - ERR should not be interpreted as a complete measure of overall hospital
      quality.
    - AI-generated insights are an interpretation layer over verified Python
      outputs and should not be treated as an independent source of facts.
    """)

st.caption(
    f"App render time: {time.time() - start:.2f} seconds"
)