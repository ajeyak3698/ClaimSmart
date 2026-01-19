import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ClaimSmart", layout="wide")

model = joblib.load('models/claim_complexity_model.pkl')

st.sidebar.title("Controls")
st.sidebar.markdown("Upload and analyze claim data")

uploaded_file = st.sidebar.file_uploader("Upload claims CSV", type=["csv"])
show_charts = st.sidebar.checkbox("Show Visual Analysis", value=True)
show_table = st.sidebar.checkbox("Show Data Table", value=True)

st.title("ClaimSmart: Intelligent Claim Routing Dashboard")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    original_df = df.copy()

    le = LabelEncoder()
    original_df['claim_type'] = le.fit_transform(original_df['claim_type'])

    X = original_df[['claim_amount', 'claim_type']]
    predictions = model.predict(X)
    actions = ['Auto-Approve' if p == 0 else ('Manual Review' if p == 2 else 'Process Normally') for p in predictions]
    original_df['Predicted Complexity'] = predictions
    original_df['Routing Action'] = actions

    st.success("File processed successfully.")

    if show_charts:
        st.subheader("Visual Analysis")

        col1, col2 = st.columns(2)
        with col1:
            action_counts = original_df['Routing Action'].value_counts()
            fig1, ax1 = plt.subplots()
            action_counts.plot(kind='bar', ax=ax1)
            ax1.set_title("Routing Action Distribution")
            ax1.set_ylabel("Number of Claims")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(original_df['claim_amount'], kde=True, bins=30, ax=ax2)
            ax2.set_title("Claim Amount Distribution")
            st.pyplot(fig2)

    if show_table:
        st.subheader("Processed Claim Data")
        st.dataframe(original_df)

    st.download_button(
        label="Download Processed CSV",
        data=original_df.to_csv(index=False).encode('utf-8'),
        file_name="processed_claims.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")
