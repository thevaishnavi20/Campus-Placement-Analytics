import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Campus Placement Analytics",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown(f"""
<style>
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #a4133c 0%, #ff4d6d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .metric-card {{
        animation: slideIn 0.5s ease-out;
    }}
    @keyframes slideIn {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    @media (max-width: 768px) {{
        .main-header {{ font-size: 2rem; }}
    }}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/campus_placement_cleaned_final.csv")

df = load_data()

# ---------------- TITLE ----------------
st.markdown('<h1 class="main-header">🎓 Campus Placement Analytics</h1>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
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

# ---------------- PLACEMENT RATE BY DEPARTMENT ----------------
st.markdown("### 🏢 Placement Rate by Department/Stream")

if 'PlacementStatus' in df_filtered.columns:
    # Create department categories based on Technical Skills Score
    df_temp = df_filtered.copy()
    if 'Technical_Skills_Score' in df_temp.columns:
        df_temp['Stream'] = pd.cut(df_temp['Technical_Skills_Score'], 
                                    bins=[0, 50, 70, 100], 
                                    labels=['Arts', 'Commerce', 'Engineering'])
    else:
        df_temp['Stream'] = 'General'
    
    dept_placement = df_temp.groupby('Stream')['PlacementStatus'].apply(
        lambda x: (x == 'placed').sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index()
    dept_placement.columns = ['Stream', 'Placement_Rate']
    
    fig = px.bar(dept_placement, x='Stream', y='Placement_Rate', 
                 title='Placement Rate by Stream (%)',
                 color='Placement_Rate',
                 color_continuous_scale=['#ffccd5', '#c9184a'])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Placement data not available")

# ---------------- TOP PERFORMERS LEADERBOARD ----------------
st.markdown("### 🏆 Top Performing Students Leaderboard")

if 'CGPA' in df_filtered.columns:
    top_students = df_filtered.nlargest(10, 'CGPA')[['CGPA']].reset_index(drop=True)
    top_students.index += 1
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(top_students.style.background_gradient(cmap='Reds'), use_container_width=True)
    with col2:
        fig = go.Figure(go.Indicator(
            mode="number",
            value=top_students['CGPA'].iloc[0] if len(top_students) > 0 else 0,
            title={'text': "🥇 Top CGPA"},
            number={'font': {'size': 50, 'color': '#c9184a'}}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("CGPA data not available for leaderboard")

# ---------------- SKILL GAP ANALYSIS ----------------
st.markdown("### 📊 Skill Gap Analysis")

skill_cols = ['Technical_Skills_Score', 'SoftSkillsRating', 'AptitudeTestScore']
available_skills = [col for col in skill_cols if col in df_filtered.columns]

if available_skills:
    skill_avg = df_filtered[available_skills].mean()
    skill_target = pd.Series([80, 4.5, 75], index=['Technical_Skills_Score', 'SoftSkillsRating', 'AptitudeTestScore'])
    
    skill_gap_data = pd.DataFrame({
        'Skill': available_skills,
        'Current Avg': [skill_avg[s] for s in available_skills],
        'Target': [skill_target[s] if s in skill_target.index else 80 for s in available_skills]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current', x=skill_gap_data['Skill'], y=skill_gap_data['Current Avg'], marker_color='#ff8fa3'))
    fig.add_trace(go.Bar(name='Target', x=skill_gap_data['Skill'], y=skill_gap_data['Target'], marker_color='#c9184a'))
    fig.update_layout(barmode='group', title='Skill Gap Analysis', height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Skill data not available")

# ---------------- TIMELINE OF PLACEMENT DRIVES ----------------
st.markdown("### 📅 Timeline of Placement Drives")

timeline_data = pd.DataFrame({
    'Company': ['TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant', 'HCL'],
    'Date': pd.date_range(start='2024-01-15', periods=6, freq='15D'),
    'Students_Placed': [45, 38, 52, 30, 41, 35]
})

fig = px.scatter(timeline_data, x='Date', y='Students_Placed', size='Students_Placed',
                 color='Company', title='Placement Drive Timeline',
                 hover_data=['Company', 'Students_Placed'])
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# ---------------- VISUALIZATIONS ----------------
st.markdown("### 📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    num_col = st.selectbox("Select numeric column", numeric_cols)
    
    fig = px.histogram(df_filtered, x=num_col, nbins=30, 
                       title=f"Distribution of {num_col}",
                       color_discrete_sequence=['#c9184a'])
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df_filtered, y=num_col, 
                 title=f"Box Plot of {num_col}",
                 color_discrete_sequence=['#ff4d6d'])
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
        marker=dict(color='#c9184a')
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
                   'bar': {'color': "#c9184a"},
                   'steps': [
                       {'range': [0, 40], 'color': "#ffccd5"},
                       {'range': [40, 70], 'color': "#ff8fa3"},
                       {'range': [70, 100], 'color': "#ff4d6d"}],
                   'threshold': {'line': {'color': "#a4133c", 'width': 4}, 'thickness': 0.75, 'value': 60}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if confidence >= 60:
            st.success(f"✅ Likely PLACED (Confidence: {confidence:.0f}%)")
        else:
            st.error(f"❌ Likely NOT PLACED (Confidence: {confidence:.0f}%)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #c9184a;'>Made with ❤️ using Streamlit | Campus Placement Analytics © 2024</div>", unsafe_allow_html=True)