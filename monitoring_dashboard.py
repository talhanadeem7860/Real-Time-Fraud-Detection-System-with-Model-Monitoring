# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import time
from scipy.stats import ks_2samp

st.set_page_config(layout="wide")
st.title("Real-Time Model Monitoring Dashboard")

# Load reference data
try:
    reference_data = pd.read_csv('reference_data.csv')
except FileNotFoundError:
    st.error("Reference data not found. Please run train_model.py first.")
    st.stop()

# Placeholders for dynamic content
placeholder = st.empty()

def calculate_drift(reference_df, live_df):
    """Calculates drift using the K-S test for numerical features."""
    drift_report = {}
    numerical_features = reference_df.select_dtypes(include=np.number).columns
    
    for feature in numerical_features:
        p_value = ks_2samp(reference_df[feature], live_df[feature]).pvalue
        drift_report[feature] = {
            'p_value': p_value,
            'drift_detected': p_value < 0.05
        }
    return drift_report

# Main monitoring loop
while True:
    try:
        live_data = pd.read_csv('live_data_log.csv')
        # Use the last N records as the current window for comparison
        current_window = live_data.tail(100)
        
        if len(current_window) < 20:
             with placeholder.container():
                st.warning("Waiting for more live data to accumulate...")
        else:
            drift_report = calculate_drift(reference_data, current_window)
            
            with placeholder.container():
                st.header("Drift Detection Report")
                
                # Overall status
                is_drifting = any(d['drift_detected'] for d in drift_report.values())
                if is_drifting:
                    st.error("DRIFT DETECTED! Model retraining may be required.")
                else:
                    st.success("No significant data drift detected. Model is healthy.")
                
                # Display detailed report
                report_df = pd.DataFrame(drift_report).T
                report_df['p_value'] = report_df['p_value'].map('{:.4f}'.format)
                st.dataframe(report_df)
                
                # Show live data summary
                st.header("Live Data Stream Summary")
                st.dataframe(current_window.describe())

    except FileNotFoundError:
        with placeholder.container():
            st.warning("Waiting for live data stream to start...")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    time.sleep(5) # Refresh rate