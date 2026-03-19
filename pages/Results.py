import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
import os

PLOTS_DIR = Path("evaluation_plots")
REPORT_PATH = Path("evaluation_report.json")
FEATURES_PATH = Path("dynamic_features.py")

def load_evaluation_report() -> dict:
    try:
        if REPORT_PATH.exists():
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def get_plots() -> list:
    try:
        if PLOTS_DIR.exists():
            return sorted(PLOTS_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
    except Exception:
        pass
    return []

def get_feature_code() -> str:
    try:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def extract_implemented_features() -> list:
    code = get_feature_code()
    if not code:
        return []
    matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', code)
    TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target', 'target_col']
    return [m for m in matches if m not in TARGET_CANDIDATES]

def show_results():
    st.header("📈 Training Results")
    
    if not st.session_state.iteration_history and not REPORT_PATH.exists():
        st.info("No training results yet. Run training first.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    report = load_evaluation_report()
    
    with col1:
        task_type = report.get("task_type", "unknown").upper()
        st.metric("Task Type", task_type)
    with col2:
        metric_name = report.get("metric_name", "Score")
        score_mean = report.get("score_mean", 0)
        st.metric(f"{metric_name} Mean", f"{score_mean:.4f}")
    with col3:
        score_std = report.get("score_std", 0)
        st.metric(f"{metric_name} Std", f"{score_std:.4f}")
    with col4:
        num_features = report.get("num_features", 0)
        st.metric("Features", num_features)
    
    st.markdown("---")
    
    tabs = st.tabs(["📊 Evaluation Report", "📉 Plots", "🔧 Features", "📝 Feature Code"])
    
    with tabs[0]:
        st.subheader("Evaluation Report")
        
        if report:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📈 Metrics")
                metrics_data = {
                    "Metric": [report.get("metric_name", "Score"), "Std Dev", "Features"],
                    "Value": [
                        f"{report.get('score_mean', 0):.4f}",
                        f"{report.get('score_std', 0):.4f}",
                        report.get("num_features", "?"),
                    ]
                }
                st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
            
            with col2:
                st.markdown("#### 🏆 Iteration History")
                if st.session_state.iteration_history:
                    history_data = []
                    for snap in st.session_state.iteration_history:
                        history_data.append({
                            "Iteration": snap.iteration,
                            "Metric": f"{snap.metric:.4f}" if snap.metric else "Failed",
                            "Status": snap.status,
                        })
                    st.dataframe(pd.DataFrame(history_data), hide_index=True)
                else:
                    st.info("No iteration history")
            
            st.markdown("#### 🔗 Top Correlations with Target")
            correlations = report.get("top_correlations_with_target", {})
            if correlations:
                corr_df = pd.DataFrame([
                    {"Feature": k, "Correlation": f"{v:.4f}"} 
                    for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                ])
                st.dataframe(corr_df, hide_index=True)
            else:
                st.info("No correlation data available")
            
            st.markdown("#### ⭐ Feature Importance")
            importance = report.get("feature_importance", {})
            if importance:
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": f"{v:.4f}"}
                    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
                ])
                st.dataframe(imp_df, hide_index=True)
            else:
                st.info("Feature importance not available")
            
            with st.expander("📄 Raw JSON Report"):
                st.json(report)
        else:
            st.warning("No evaluation report found. Run training to generate one.")
    
    with tabs[1]:
        st.subheader("Evaluation Plots")
        
        plots = get_plots()
        if plots:
            st.info(f"Found {len(plots)} plots")
            
            cols = st.columns(2)
            for i, plot_path in enumerate(plots[:10]):
                with cols[i % 2]:
                    st.image(str(plot_path), caption=plot_path.name, use_container_width=True)
            
            if len(plots) > 10:
                st.info(f"Showing 10 of {len(plots)} plots. More available in `evaluation_plots/` folder.")
        else:
            st.info("No plots available. Run training to generate plots.")
    
    with tabs[2]:
        st.subheader("Implemented Features")
        
        features = extract_implemented_features()
        if features:
            st.success(f"Found {len(features)} engineered features:")
            for feat in features:
                st.markdown(f"- `{feat}`")
        else:
            st.info("No engineered features found. The baseline model is being used.")
        
        if st.session_state.iteration_history:
            best = max(
                [s for s in st.session_state.iteration_history if s.metric is not None],
                key=lambda x: x.metric,
                default=None
            )
            if best:
                st.markdown(f"#### 🏆 Best Iteration: {best.iteration}")
                st.metric("Best Metric", f"{best.metric:.4f}")
    
    with tabs[3]:
        st.subheader("Generated Feature Engineering Code")
        
        code = get_feature_code()
        if code:
            st.code(code, language="python")
            
            st.download_button(
                label="📥 Download dynamic_features.py",
                data=code,
                file_name="dynamic_features.py",
                mime="text/x-python",
            )
        else:
            st.info("No feature engineering code found.")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Results", use_container_width=True):
        st.rerun()
