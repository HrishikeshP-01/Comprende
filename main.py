import os
import io
#import json
import time
#import uuid
import math
from typing import TypedDict, List, Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from tooling.tidb import *
from tooling.vector_search import *
#from tooling.agent_capabilities import *
from agent import *


load_dotenv()
os.environ["OPENAI_API_KEY"] =os.getenv('OPENAI_API_KEY')


st.set_page_config(page_title="LangGraph + TiDB Classroom Evaluator", layout="wide")
def main():
    st.title("📘Comprende: LangGraph + TiDB Comprehension Evaluator")
    homework_table = os.getenv("VECTOR_TABLE", "homework_vector")
    lesson_table = os.getenv("LESSON_VECTOR_TABLE", "lesson_vector")

    # Ensure relational schema
    engine = get_engine()
    ensure_relational_schema(engine)

    st.sidebar.header("Setup")
    student_names_str = st.sidebar.text_area("Student names (comma-separated)", "Aarav,Bhavana,Chitra,Dev")
    student_names = [s.strip() for s in student_names_str.split(",") if s.strip()]
    concepts_str = st.sidebar.text_area("Concepts to evaluate (comma-separated)", "Binary Search,Recursion,Dynamic Programming,Graph Traversal")
    concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]
    st.sidebar.write("---")
    st.sidebar.subheader("Vector Tables")
    st.sidebar.text_input("Homework vector table", homework_table, key="hw_table")
    st.sidebar.text_input("Lesson vector table", lesson_table, key="lsn_table")

    col1, col2 = st.columns(2)

    with col1:
        st.header("1) Ingest student homework PDFs → TiDB Vector")
        st.caption("Upload PDFs and assign to a student. They will be chunked, embedded and stored in TiDB Vector with metadata (student_name, source_file).")
        uploaded_homework = st.file_uploader("Upload homework PDFs", type=["pdf"], accept_multiple_files=True, key="hw_upload")
        selected_student = st.selectbox("Assign uploaded files to student", options=student_names, index=0 if student_names else None)
        if st.button("Ingest selected PDFs to TiDB Vector (Homework)"):
            if not uploaded_homework or not selected_student:
                st.error("Please upload at least one PDF and select a student.")
            else:
                vs = get_vectorstore(st.session_state.get("hw_table", homework_table))
                all_docs = []
                for f in uploaded_homework:
                    docs = load_pdfs_to_docs(f, selected_student)
                    all_docs.extend(docs)
                ingest_documents(all_docs, vs)
                st.success(f"Ingested {len(all_docs)} chunks for {selected_student} into TiDB Vector.")

    with col2:
        st.header("2) Ingest lesson/reference PDFs → TiDB Vector")
        st.caption("Upload lesson materials. These will be used to ground lesson plans and personalized homework.")
        uploaded_lessons = st.file_uploader("Upload lesson/reference PDFs", type=["pdf"], accept_multiple_files=True, key="lsn_upload")
        if st.button("Ingest selected PDFs to TiDB Vector (Lessons)"):
            if not uploaded_lessons:
                st.error("Please upload at least one PDF.")
            else:
                vs = get_vectorstore(st.session_state.get("lsn_table", lesson_table))
                all_docs = []
                for f in uploaded_lessons:
                    # For lesson docs, we keep student_name = 'LESSON'
                    docs = load_pdfs_to_docs(f, student_name="LESSON")
                    all_docs.extend(docs)
                ingest_documents(all_docs, vs)
                st.success(f"Ingested {len(all_docs)} lesson chunks into TiDB Vector.")

    st.markdown("---")

    st.header("3) Run end-to-end agent (LangGraph)")
    st.caption("Runs: evaluate → report → knowledge graph + groups → lesson plans → personalized homework")
    if st.button("Run full pipeline now"):
        graph = build_graph()
        init: PipelineState = {
            "concepts": concepts,
            "students": student_names,
            "homework_vector_table": st.session_state.get("hw_table", homework_table),
            "lesson_vector_table": st.session_state.get("lsn_table", lesson_table),
        }
        with st.status("Running agent… This can take a few minutes depending on PDFs & model.", expanded=True):
            out = graph.invoke(init)
        st.success("Agent run completed.")

        # Cache to session
        st.session_state["scores"] = out.get("scores", {})
        st.session_state["pain_points"] = out.get("pain_points", {})
        st.session_state["reports"] = out.get("reports", {})
        st.session_state["groups"] = out.get("groups", {})
        st.session_state["lesson_plans"] = out.get("lesson_plans", {})
        st.session_state["homework"] = out.get("homework", {})

    st.markdown("---")
    st.header("4) Results & Visuals")


    if "reports" in st.session_state and st.session_state["reports"]:
        st.subheader("Per-student reports")
        for s, md in st.session_state["reports"].items():
            with st.expander(f"Report: {s}", expanded=False):
                st.markdown(md)

    if "scores" in st.session_state and st.session_state["scores"]:
        st.subheader("Knowledge graph (students ↔ concepts they are aware of)")
        fig = plot_knowledge_graph(st.session_state["scores"], concepts, aware_threshold=70)
        st.pyplot(fig)

    if "groups" in st.session_state and st.session_state["groups"]:
        st.subheader("Study groups")
        for gid, members in st.session_state["groups"].items():
            st.write(f"**Group {gid}**: {', '.join(members)}")

    if "lesson_plans" in st.session_state and st.session_state["lesson_plans"]:
        st.subheader("Lesson plans (weak concepts)")
        for c, plan in st.session_state["lesson_plans"].items():
            with st.expander(f"Lesson: {c}", expanded=False):
                st.markdown(plan)

    if "homework" in st.session_state and st.session_state["homework"]:
        st.subheader("Personalized homework")
        for s, hw in st.session_state["homework"].items():
            with st.expander(f"{s}", expanded=False):
                st.markdown(hw)

if __name__ == "__main__":
    main()