import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Campus Placement Analytics Dashboard",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/campus_placement_cleaned_final.csv")

df = load_data()

# ---------------- TITLE ----------------
st.markdown("## 🎓 Campus Placement Analytics Dashboard")
st.success("CSV file loaded successfully!")

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
st.markdown("### 📌 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Total Students", df_filtered.shape[0])
c2.metric("Total Columns", df_filtered.shape[1])
c3.metric("Unique Values", df_filtered[filter_col].nunique())

# ---------------- DATASET PREVIEW ----------------
st.markdown("### 📋 Dataset Preview")
st.dataframe(df_filtered.head(10), use_container_width=True)

# ---------------- NUMERIC COLUMN VISUAL ----------------
st.markdown("### 📊 Numeric Column Visualization")

numeric_cols = df_filtered.select_dtypes(include=np.number).columns
num_col = st.selectbox("Select numeric column", numeric_cols)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_filtered[num_col].values)
ax.set_title(num_col)
ax.set_ylabel("Value")
ax.set_xlabel("Index")
fig.patch.set_facecolor("none")

st.pyplot(fig)

# ---------------- PLACEMENT PREDICTION ----------------
st.markdown("### 🧠 Placement Prediction")

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

display_df = pd.DataFrame({
    "Feature": features,
    "Value": student[features].values
})

st.table(display_df)

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

if st.button("Predict Placement"):
    confidence = predict_placement(student)

    if confidence >= 60:
        st.success(f"✅ Likely PLACED (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"❌ Likely NOT PLACED (Confidence: {confidence:.2f}%)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")