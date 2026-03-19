import streamlit as st
import pandas as pd
from pathlib import Path

def show_home():
    st.header("🏠 Welcome to AutoML Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset", "✅ Loaded" if st.session_state.dataset_uploaded else "❌ Missing")
        if st.session_state.dataset_info:
            st.caption(f"Rows: {st.session_state.dataset_info['rows']}")
            st.caption(f"Columns: {st.session_state.dataset_info['columns']}")
    
    with col2:
        glossary_status = "✅ Defined" if st.session_state.glossary_content else "❌ Empty"
        st.metric("Glossary", glossary_status)
        if st.session_state.glossary_content:
            st.caption(f"Characters: {len(st.session_state.glossary_content)}")
    
    with col3:
        st.metric("Iterations", st.session_state.max_iterations)
        st.caption(f"MLFlow: {'Enabled' if st.session_state.mlflow_enabled else 'Disabled'}")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📋 Overview", "📄 Current Data"])
    
    with tab1:
        st.subheader("How it works")
        
        st.markdown("""
        ### 1️⃣ Prepare Your Data
        - Upload your CSV dataset using the sidebar
        - Define the glossary with feature descriptions and business context
        
        ### 2️⃣ Configure Training
        - Set the number of iterations
        - Enable/disable MLFlow tracking
        - Configure experiment name
        
        ### 3️⃣ Start Training
        - Go to **Training** page and click **Start Training**
        - Monitor progress in real-time
        
        ### 4️⃣ View Results
        - Check **Results** page for evaluation reports
        - View generated plots and feature importance
        """)
        
        st.markdown("---")
        
        ready = st.session_state.dataset_uploaded and st.session_state.glossary_content
        
        if ready:
            st.success("🎉 All set! You're ready to start training.")
            if st.button("🚀 Go to Training Page", type="primary", use_container_width=True):
                st.session_state.page = "Training"
                st.rerun()
        else:
            missing = []
            if not st.session_state.dataset_uploaded:
                missing.append("Dataset")
            if not st.session_state.glossary_content:
                missing.append("Glossary")
            st.warning(f"⚠️ Please complete: {', '.join(missing)}")
    
    with tab2:
        if st.session_state.dataset_info:
            try:
                df = pd.read_csv("data/dataset.csv", encoding="latin-1")
                
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.subheader("Column Info")
                col_info = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.count(),
                    "Null": df.isnull().sum(),
                    "Unique": df.nunique(),
                })
                st.dataframe(col_info, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        else:
            st.info("No dataset uploaded yet. Please upload a CSV file in the sidebar.")
        
        st.markdown("---")
        
        if st.session_state.glossary_content:
            st.subheader("Glossary Preview")
            st.markdown(st.session_state.glossary_content)
