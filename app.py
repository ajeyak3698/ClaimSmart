import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("Anomaly Detection in Time-Series Data")

st.sidebar.header("Settings")

np.random.seed(42)
n_points = 500
data = np.random.normal(loc=50, scale=5, size=n_points)
data[100], data[250], data[400] = 100, 120, 90

df = pd.DataFrame({
    'timestamp': pd.date_range("2023-01-01", periods=n_points, freq='H'),
    'value': data
})

contamination = st.sidebar.slider("Contamination Rate", 0.001, 0.1, 0.02, step=0.001)
show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)
date_range = st.sidebar.date_input(
    "Filter by Date",
    [df['timestamp'].min().date(), df['timestamp'].max().date()]
)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['value']])

clf = IForest(contamination=contamination)
clf.fit(X_scaled)
df['anomaly'] = clf.predict(X_scaled)

st.subheader("Time-Series Data")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['timestamp'], df['value'], label="Value", linewidth=1)
if show_anomalies:
    ax.scatter(
        df[df['anomaly'] == 1]['timestamp'],
        df[df['anomaly'] == 1]['value'],
        color='red',
        label="Anomaly",
        s=30
    )
ax.set_title("Detected Anomalies")
ax.legend()
st.pyplot(fig)

st.subheader("Value Distribution")
fig2, ax2 = plt.subplots()
df['value'].hist(bins=30, ax=ax2)
ax2.set_title("Histogram of Values")
st.pyplot(fig2)

st.subheader("Value Boxplot")
fig3, ax3 = plt.subplots()
ax3.boxplot(df['value'], vert=False)
ax3.set_title("Boxplot of Values")
st.pyplot(fig3)

st.subheader("Anomaly Summary")
anomaly_count = df['anomaly'].sum()
st.metric("Anomaly Count", anomaly_count)

csv = df.to_csv(index=False).encode()
st.download_button("Download Results", csv, "anomaly_results.csv", "text/csv")
