import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Campus Placement Analytics Dashboard",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #9c89b8 0%, #f0a6ca 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/campus_placement_cleaned_final.csv")

df = load_data()

# ---------------- TITLE ----------------
st.markdown('<h1 class="main-header">🎓 Campus Placement Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("🔎 Filter Data")

filter_col = st.sidebar.selectbox("Select column", df.columns)

selected_values = st.sidebar.multiselect(
    "Select values",
    df[filter_col].unique()
)

if selected_values:
    df_filtered = df[df[filter_col].isin(selected_values)]
else:
    df_filtered = df.copy()

# ---------------- KEY METRICS ----------------
st.markdown("### 📊 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📚 Total Students", df_filtered.shape[0], delta="Active")
with c2:
    placed = df_filtered[df_filtered.get('PlacementStatus', df_filtered.columns[0]) == 'Placed'].shape[0] if 'PlacementStatus' in df_filtered.columns else 0
    st.metric("✅ Placed", placed)
with c3:
    avg_score = df_filtered.select_dtypes(include=np.number).mean().mean()
    st.metric("📈 Avg Score", f"{avg_score:.1f}")
with c4:
    st.metric("🔍 Filtered By", filter_col[:15])

# ---------------- DATASET PREVIEW ----------------
st.markdown("### 📋 Dataset Preview")
st.dataframe(df_filtered.head(10), use_container_width=True)

# ---------------- VISUALIZATIONS ----------------
st.markdown("### 📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    num_col = st.selectbox("Select numeric column", numeric_cols)
    
    fig = px.histogram(df_filtered, x=num_col, nbins=30, 
                       title=f"Distribution of {num_col}",
                       color_discrete_sequence=['#9c89b8'])
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df_filtered, y=num_col, 
                 title=f"Box Plot of {num_col}",
                 color_discrete_sequence=['#f0a6ca'])
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- PLACEMENT PREDICTION ----------------
st.markdown("---")
st.markdown("### 🎯 Placement Prediction")

col1, col2 = st.columns([1, 2])

with col1:
    student_index = st.selectbox(
        "Select Student Index",
        df_filtered.index
    )
    student = df_filtered.loc[student_index]
    
    features = [
        "Internships",
        "Projects",
        "Workshops/Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "ExtracurricularActivities",
        "PlacementTraining",
        "SSC_Marks",
        "HSC_Marks",
        "Technical_Skills_Score"
    ]
    
    available_features = [f for f in features if f in student.index]
    
    display_df = pd.DataFrame({
        "Feature": available_features,
        "Value": [student[f] for f in available_features]
    })
    
    st.dataframe(display_df, use_container_width=True, height=400)

with col2:
    fig = go.Figure(go.Bar(
        x=[student[f] for f in available_features if pd.api.types.is_numeric_dtype(type(student[f]))],
        y=[f for f in available_features if pd.api.types.is_numeric_dtype(type(student[f]))],
        orientation='h',
        marker=dict(color='#9c89b8')
    ))
    fig.update_layout(title="Student Profile", height=400, xaxis_title="Score", yaxis_title="Features")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- SIMPLE PREDICTION LOGIC ----------------
def predict_placement(row):
    score = 0

    if row["Internships"] > 0:
        score += 15
    if row["Projects"] >= 2:
        score += 15
    if row["Workshops/Certifications"] >= 1:
        score += 10
    if row["AptitudeTestScore"] >= 70:
        score += 15
    if row["SoftSkillsRating"] >= 4:
        score += 10
    if row["Technical_Skills_Score"] >= 60:
        score += 15
    if row["PlacementTraining"] == "Yes":
        score += 10

    confidence = min(score, 100)
    return confidence

if st.button("🔮 Predict Placement", use_container_width=True):
    confidence = predict_placement(student)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            title={'text': "Placement Confidence"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#9c89b8"},
                   'steps': [
                       {'range': [0, 40], 'color': "#f0e6ef"},
                       {'range': [40, 70], 'color': "#efc3e6"},
                       {'range': [70, 100], 'color': "#f0a6ca"}],
                   'threshold': {'line': {'color': "#9c89b8", 'width': 4}, 'thickness': 0.75, 'value': 60}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if confidence >= 60:
            st.success(f"✅ Likely PLACED (Confidence: {confidence:.0f}%)")
        else:
            st.error(f"❌ Likely NOT PLACED (Confidence: {confidence:.0f}%)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #9c89b8;'>Made with ❤️ using Streamlit | Campus Placement Analytics © 2024</div>", unsafe_allow_html=True)