import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        with open('fraud_detection_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'fraud_detection_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

# Main title and description
st.title("🔍 Fraud Detection System")
st.markdown("""
This system uses machine learning to detect fraudulent credit card transactions.
Upload a CSV file with transaction data or enter transaction details manually.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose input method:", 
                         ["Manual Input", "CSV File Upload"])

# Load model and scaler
model, scaler = load_model()

if model is not None and scaler is not None:
    if option == "Manual Input":
        st.header("Manual Transaction Input")
        
        # Create input fields for transaction features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
            v1 = st.number_input("V1", value=0.0, step=0.1)
            v2 = st.number_input("V2", value=0.0, step=0.1)
            v3 = st.number_input("V3", value=0.0, step=0.1)
            v4 = st.number_input("V4", value=0.0, step=0.1)
            
        with col2:
            v5 = st.number_input("V5", value=0.0, step=0.1)
            v6 = st.number_input("V6", value=0.0, step=0.1)
            v7 = st.number_input("V7", value=0.0, step=0.1)
            v8 = st.number_input("V8", value=0.0, step=0.1)
            v9 = st.number_input("V9", value=0.0, step=0.1)
            
        with col3:
            v10 = st.number_input("V10", value=0.0, step=0.1)
            v11 = st.number_input("V11", value=0.0, step=0.1)
            v12 = st.number_input("V12", value=0.0, step=0.1)
            v13 = st.number_input("V13", value=0.0, step=0.1)
            v14 = st.number_input("V14", value=0.0, step=0.1)
        
        # Add more features if needed
        col4, col5, col6 = st.columns(3)
        
        with col4:
            v15 = st.number_input("V15", value=0.0, step=0.1)
            v16 = st.number_input("V16", value=0.0, step=0.1)
            v17 = st.number_input("V17", value=0.0, step=0.1)
            
        with col5:
            v18 = st.number_input("V18", value=0.0, step=0.1)
            v19 = st.number_input("V19", value=0.0, step=0.1)
            v20 = st.number_input("V20", value=0.0, step=0.1)
            
        with col6:
            v21 = st.number_input("V21", value=0.0, step=0.1)
            v22 = st.number_input("V22", value=0.0, step=0.1)
            v23 = st.number_input("V23", value=0.0, step=0.1)
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            v24 = st.number_input("V24", value=0.0, step=0.1)
            v25 = st.number_input("V25", value=0.0, step=0.1)
            
        with col8:
            v26 = st.number_input("V26", value=0.0, step=0.1)
            v27 = st.number_input("V27", value=0.0, step=0.1)
            
        with col9:
            v28 = st.number_input("V28", value=0.0, step=0.1)
        
        # Predict button
        if st.button("Detect Fraud"):
            # Create feature array
            features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                                v21, v22, v23, v24, v25, v26, v27, v28, amount]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            prediction_proba = model.predict_proba(features_scaled)
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction[0] == 1:
                st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED**")
                st.metric("Fraud Probability", f"{prediction_proba[0][1]*100:.2f}%")
                st.warning("This transaction has been flagged as potentially fraudulent. Please review carefully.")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
                st.metric("Fraud Probability", f"{prediction_proba[0][1]*100:.2f}%")
                st.info("This transaction appears to be legitimate.")
            
            # Show confidence scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Probability", f"{prediction_proba[0][0]*100:.2f}%")
            with col2:
                st.metric("Fraud Probability", f"{prediction_proba[0][1]*100:.2f}%")
    
    else:  # CSV File Upload
        st.header("Batch Processing via CSV Upload")
        
        uploaded_file = st.file_uploader("Upload CSV file with transaction data", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Check if required columns are present
                required_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                else:
                    if st.button("Process Transactions"):
                        # Prepare features
                        features = df[required_columns]
                        
                        # Scale features
                        features_scaled = scaler.transform(features)
                        
                        # Make predictions
                        predictions = model.predict(features_scaled)
                        predictions_proba = model.predict_proba(features_scaled)
                        
                        # Add predictions to dataframe
                        df['Prediction'] = predictions
                        df['Fraud_Probability'] = predictions_proba[:, 1]
                        df['Legitimate_Probability'] = predictions_proba[:, 0]
                        df['Status'] = df['Prediction'].apply(lambda x: 'Fraudulent' if x == 1 else 'Legitimate')
                        
                        # Display results
                        st.subheader("Detection Results")
                        
                        # Summary statistics
                        fraud_count = df['Prediction'].sum()
                        total_count = len(df)
                        fraud_percentage = (fraud_count / total_count) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", total_count)
                        with col2:
                            st.metric("Fraudulent Transactions", fraud_count)
                        with col3:
                            st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
                        
                        # Display results table
                        st.dataframe(df[['Amount', 'Status', 'Fraud_Probability', 'Legitimate_Probability']])
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
                        
                        # Show high-risk transactions
                        high_risk = df[df['Fraud_Probability'] > 0.7]
                        if not high_risk.empty:
                            st.subheader("🚨 High-Risk Transactions (Probability > 70%)")
                            st.dataframe(high_risk[['Amount', 'Fraud_Probability']].sort_values('Fraud_Probability', ascending=False))
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

else:
    st.warning("""
    **Setup Required:**
    
    To use this fraud detection system, you need to train and save the model first.
    
    Please run the following code to train and save the model:
    
    ```python
    # Add this to your notebook after training
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    # Assuming you have X_train, X_test, y_train, y_test
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save model and scaler
    with open('fraud_detection_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    ```
    """)

# Add information section
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This fraud detection system uses machine learning to identify potentially fraudulent credit card transactions.

**Features used:**
- 28 anonymized features (V1-V28)
- Transaction amount

**Model:** Random Forest Classifier
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Fraud Detection System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
