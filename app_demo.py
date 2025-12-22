# app_demo.py
import streamlit as st
import graphviz
from main import run_pipeline # <-- SỬA ĐỔI QUAN TRỌNG
from src.utils.neo4j_connect import db_connector

st.set_page_config(page_title="MedCOT Demo", layout="wide")
st.title("🧠 MedCOT: Neuro-Symbolic Medical AI")

# Sidebar
with st.sidebar:
    st.header("Settings")
    # use_gnn và check_safety giờ được điều khiển trong pipeline
    # Bạn có thể thêm các config vào hàm run_pipeline nếu muốn
    if db_connector:
        st.success("✅ Neo4j Connected")
    else:
        st.error("❌ Neo4j Disconnected")

# Main Input
query = st.text_input("Enter Medical Question:", "Is it safe to take Warfarin and Aspirin together?")

if st.button("Run Analysis"):
    if not db_connector:
        st.error("Database connection required.")
        st.stop()
        
    with st.spinner("Running MedCOT Pipeline... This may take a moment."):
        try:
            # Gọi pipeline đã được chuẩn hóa từ main.py
            state = run_pipeline(query=query)

            if state is None:
                st.error("Pipeline execution failed. Check logs for details.")
                st.stop()

            # Display Results
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("💡 Final Answer")
                st.markdown(state.final_answer)
                if state.safety_flags:
                    st.error(f"🚨 {len(state.safety_flags)} Safety Alerts Detected!")
                    for flag in state.safety_flags:
                        st.warning(flag['msg'])
            
            with col2:
                st.subheader("🕸 Reasoning Graph")
                if state.verified_path:
                    graph = graphviz.Digraph()
                    graph.attr(rankdir='LR')
                    
                    node_map = {n['id']: n['name'] for n in state.graph_refs.get("ckg_subgraph", {}).get("nodes", [])}

                    for step in state.verified_path:
                        s_name = node_map.get(step['source'], str(step['source']))[:25]
                        t_name = node_map.get(step['target'], str(step['target']))[:25]
                        edge_label = step.get('edge_text', step.get('edge', 'rel'))

                        graph.node(s_name, style='filled', fillcolor='lightblue')
                        graph.node(t_name, style='filled', fillcolor='lightgreen')
                        graph.edge(s_name, t_name, label=edge_label)
                    st.graphviz_chart(graph)
                else:
                    st.warning("No verified reasoning path was generated.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")