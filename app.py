import streamlit as st
import os
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="AutoML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PLOTS_DIR = Path("evaluation_plots")
GLOSSARY_PATH = Path("glossary.md")
DATASET_PATH = DATA_DIR / "dataset.csv"
REPORT_PATH = Path("evaluation_report.json")


def init_session_state():
    if "training_runner" not in st.session_state:
        st.session_state.training_runner = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "iteration_history" not in st.session_state:
        st.session_state.iteration_history = []
    if "training_status" not in st.session_state:
        st.session_state.training_status = "idle"
    if "current_iteration" not in st.session_state:
        st.session_state.current_iteration = 0
    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = 5
    if "mlflow_experiment" not in st.session_state:
        st.session_state.mlflow_experiment = "AutoLLml_Experiments"
    if "mlflow_enabled" not in st.session_state:
        st.session_state.mlflow_enabled = True
    if "glossary_content" not in st.session_state:
        st.session_state.glossary_content = ""
    if "dataset_uploaded" not in st.session_state:
        st.session_state.dataset_uploaded = DATASET_PATH.exists()
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None
        if DATASET_PATH.exists():
            try:
                import pandas as pd
                df = pd.read_csv(DATASET_PATH, encoding="latin-1")
                st.session_state.dataset_info = {
                    "filename": "dataset.csv",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "dtypes": str(df.dtypes.to_dict()),
                }
            except Exception:
                pass
    if "business_strategy" not in st.session_state:
        st.session_state.business_strategy = None


def save_glossary(content: str) -> bool:
    try:
        with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        st.session_state.glossary_content = content
        return True
    except Exception as e:
        st.error(f"Failed to save glossary: {e}")
        return False


def save_dataset(uploaded_file) -> bool:
    try:
        DATA_DIR.mkdir(exist_ok=True)
        with open(DATASET_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.dataset_uploaded = True
        import pandas as pd
        df = pd.read_csv(DATASET_PATH, encoding="latin-1")
        st.session_state.dataset_info = {
            "filename": uploaded_file.name,
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": str(df.dtypes.to_dict()),
        }
        return True
    except Exception as e:
        st.error(f"Failed to save dataset: {e}")
        return False


def load_evaluation_report() -> dict:
    try:
        if REPORT_PATH.exists():
            import json
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
        with open("dynamic_features.py", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


init_session_state()

st.title("🤖 AutoML Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📊 Dataset", expanded=True):
        if st.session_state.dataset_uploaded and st.session_state.dataset_info:
            st.success(f"✅ {st.session_state.dataset_info['filename']}")
            st.caption(f"Rows: {st.session_state.dataset_info['rows']} | Columns: {st.session_state.dataset_info['columns']}")
        else:
            st.warning("No dataset uploaded")
        
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Saving..."):
                if save_dataset(uploaded_file):
                    st.success("Dataset saved!")
                    st.rerun()
    
    st.markdown("---")
    
    with st.expander("📝 Glossary", expanded=True):
        if GLOSSARY_PATH.exists():
            default_glossary = GLOSSARY_PATH.read_text(encoding="utf-8")
        else:
            default_glossary = st.session_state.glossary_content or "# Glossary\n\n- `target`: Target variable\n- `feature1`: Description"
        
        glossary_editor = st.text_area(
            "Edit Glossary (Markdown)",
            value=default_glossary,
            height=200,
            key="glossary_editor"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", type="primary"):
                if save_glossary(glossary_editor):
                    st.success("Glossary saved!")
        with col2:
            if st.button("🔄 Load"):
                st.rerun()
    
    st.markdown("---")
    
    with st.expander("🔧 Training Parameters", expanded=True):
        st.session_state.max_iterations = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=50,
            value=st.session_state.max_iterations,
        )
        
        st.session_state.mlflow_experiment = st.text_input(
            "MLFlow Experiment Name",
            value=st.session_state.mlflow_experiment,
        )
        
        st.session_state.mlflow_enabled = st.checkbox(
            "Enable MLFlow Tracking",
            value=st.session_state.mlflow_enabled,
        )
    
    st.markdown("---")
    
    status_color = {
        "idle": "gray",
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "stopped": "orange",
    }.get(st.session_state.training_status, "gray")
    
    st.markdown(f"**Status:** :{status_color}[{st.session_state.training_status.upper()}]")
    
    if st.session_state.training_status == "running" and st.session_state.training_runner:
        if st.button("⏹️ Stop Training", type="secondary"):
            st.session_state.training_runner.stop()
            st.session_state.training_status = "stopped"
            st.rerun()

st.markdown("---")

pages = {
    "🏠 Home": "Home",
    "📊 Training": "Training",
    "📈 Results": "Results",
}

page = st.radio("Navigate", list(pages.keys()), horizontal=True, label_visibility="collapsed")

if page == "🏠 Home":
    from pages.Home import show_home
    show_home()
elif page == "📊 Training":
    from pages.Training import show_training
    show_training()
elif page == "📈 Results":
    from pages.Results import show_results
    show_results()
