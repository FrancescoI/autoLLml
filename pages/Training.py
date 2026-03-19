import streamlit as st
import time
from runner import start_training_thread

def show_training():
    st.header("📊 Training Dashboard")
    
    ready = st.session_state.dataset_uploaded and st.session_state.glossary_content
    
    if not ready:
        st.warning("⚠️ Please upload dataset and glossary first in the sidebar.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Iterations", f"{st.session_state.current_iteration}/{st.session_state.max_iterations}")
    with col2:
        if st.session_state.iteration_history:
            latest = st.session_state.iteration_history[-1]
            if latest.metric:
                st.metric("Latest Metric", f"{latest.metric:.4f}")
            else:
                st.metric("Latest Metric", "—")
        else:
            st.metric("Latest Metric", "—")
    with col3:
        completed = len([s for s in st.session_state.iteration_history if s.metric is not None])
        st.metric("Completed", completed)
    with col4:
        status = st.session_state.training_status.upper()
        color = {"IDLE": "gray", "RUNNING": "blue", "COMPLETED": "green", "FAILED": "red", "STOPPED": "orange"}.get(status, "gray")
        st.markdown(f"**Status:** :{color}[{status}]")
    
    if st.session_state.training_status == "running":
        progress = st.session_state.current_iteration / st.session_state.max_iterations
        st.progress(progress, text=f"Running iteration {st.session_state.current_iteration}/{st.session_state.max_iterations}...")
    elif st.session_state.training_status == "completed":
        st.success("✅ Training completed!")
    elif st.session_state.training_status == "failed":
        st.error("❌ Training failed")
    
    st.markdown("---")
    
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("📝 Training Logs")
        
        log_container = st.container()
        
        with log_container:
            if st.session_state.logs:
                for log in st.session_state.logs[-100:]:
                    if "[ERROR]" in log or "[!]" in log:
                        st.error(log)
                    elif "[+]" in log or "completed" in log.lower():
                        st.success(log)
                    elif "[*]" in log:
                        st.info(log)
                    else:
                        st.text(log)
            else:
                st.info("No logs yet. Start training to see logs here.")
        
        if st.session_state.training_status == "running":
            st.empty()
            time.sleep(2)
            st.rerun()
    
    with right_col:
        st.subheader("📈 Iteration History")
        
        if st.session_state.iteration_history:
            for snap in st.session_state.iteration_history:
                with st.container():
                    if snap.metric is not None:
                        st.metric(f"Iteration {snap.iteration}", f"{snap.metric:.4f}", delta=f"{snap.status}")
                    else:
                        st.metric(f"Iteration {snap.iteration}", "Failed", delta=f"{snap.status}")
                    st.caption(snap.timestamp.strftime("%H:%M:%S"))
                    st.divider()
        else:
            st.info("No iterations yet.")
    
    st.markdown("---")
    
    control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
    
    with control_col2:
        if st.session_state.training_status in ["idle", "stopped", "completed", "failed"]:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.session_state.training_status = "running"
                st.session_state.logs = []
                st.session_state.iteration_history = []
                st.session_state.current_iteration = 0
                st.session_state.training_runner = start_training_thread(
                    max_iterations=st.session_state.max_iterations,
                    mlflow_experiment_name=st.session_state.mlflow_experiment,
                    mlflow_tracking_enabled=st.session_state.mlflow_enabled,
                )
                time.sleep(1)
                st.rerun()
        elif st.session_state.training_status == "running":
            if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                if st.session_state.training_runner:
                    st.session_state.training_runner.stop()
                st.session_state.training_status = "stopped"
                st.rerun()
    
    if st.session_state.training_status == "running" and st.session_state.training_runner:
        new_logs = st.session_state.training_runner.get_logs()
        if new_logs:
            st.session_state.logs.extend(new_logs)
        
        new_history = st.session_state.training_runner.get_history()
        if len(new_history) > len(st.session_state.iteration_history):
            st.session_state.iteration_history = new_history
            st.session_state.current_iteration = len(new_history)
        
        if not st.session_state.training_runner.is_running and st.session_state.training_status == "running":
            if new_history:
                failed_count = len([s for s in new_history if s.status == "failed"])
                if failed_count == len(new_history):
                    st.session_state.training_status = "failed"
                else:
                    st.session_state.training_status = "completed"
            else:
                st.session_state.training_status = "failed"
