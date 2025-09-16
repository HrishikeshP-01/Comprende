from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from tooling.agent_capabilities import *
from tooling.vector_search import *
import streamlit as st

class PipelineState(TypedDict, total=False):
    concepts: List[str]
    students: List[str]
    homework_vector_table: str
    lesson_vector_table: str
    scores: Dict[str, Dict[str, float]]
    pain_points: Dict[str, Dict[str, List[str]]]
    reports: Dict[str, str]
    groups: Dict[int, List[str]]
    lesson_plans: Dict[str, str]
    homework: Dict[str, str]

def node_ingest(state: PipelineState) -> PipelineState:
    # ingestion is driven by UI for now(files are uploaded & added to vector store).
    return state

def node_evaluate(state: PipelineState) -> PipelineState:
    llm = get_llm()
    vs = get_vectorstore(state["homework_vector_table"])
    engine = get_engine()
    scores: Dict[str, Dict[str, float]] = {}
    pain_points: Dict[str, Dict[str, List[str]]] = {}
    for student in state["students"]:
        st.write(f"Evaluating **{student}** …")
        scores[student] = {}
        pain_points[student] = {}
        for concept in state["concepts"]:
            docs = retrieve_student_context(vs, student, concept, k=6)
            snippets = [f"[{(d.metadata or {}).get('source_file','?')}] {d.page_content[:600]}" for d in docs]
            result = score_comprehension(llm, student, concept, snippets)
            scr = float(result.get("score", 0))
            pts = result.get("pain_points", [])
            scores[student][concept] = scr
            pain_points[student][concept] = pts
            write_comprehension(engine, student, concept, scr, pts)
    state["scores"] = scores
    state["pain_points"] = pain_points
    return state

def node_reports(state: PipelineState) -> PipelineState:
    llm = get_llm()
    engine = get_engine()
    reports: Dict[str, str] = {}
    for student in state["students"]:
        results_json = json.dumps({
            "scores": state["scores"][student],
            "pain_points": state["pain_points"][student]
        }, indent=2)
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = [SystemMessage(content=SYSTEM_REPORT),
                HumanMessage(content=USER_REPORT.format(student_name=student, results_json=results_json))]
        md = llm.invoke(msgs).content
        reports[student] = md
    state["reports"] = reports
    return state

def node_knowledge_graph_and_groups(state: PipelineState) -> PipelineState:
    engine = get_engine()
    upsert_students_and_concepts(engine, state["students"], state["concepts"])
    write_student_concepts(engine, state["scores"])
    groups = build_study_groups(state["scores"], state["concepts"], target_size=2)
    write_study_groups(engine, groups)
    state["groups"] = groups
    return state

def node_lesson_plans(state: PipelineState) -> PipelineState:
    llm = get_llm()
    vs_lessons = get_vectorstore(state["lesson_vector_table"])
    weak = []
    for c in state["concepts"]:
        vals = [state["scores"][s].get(c, 0) for s in state["students"]]
        vals.sort()
        med = vals[len(vals)//2]
        if med < 70:
            weak.append(c)
    plans: Dict[str, str] = {}
    for c in weak:
        ctx_docs = vs_lessons.similarity_search(c, k=6)
        ctx_text = "\n---\n".join([d.page_content[:600] for d in ctx_docs])
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = [SystemMessage(content=SYSTEM_LESSON),
                HumanMessage(content=USER_LESSON.format(weak_concepts=", ".join([c]), context=ctx_text))]
        plan = llm.invoke(msgs).content
        plans[c] = plan
    write_lesson_plans(get_engine(), plans)
    state["lesson_plans"] = plans
    return state

def node_homework(state: PipelineState) -> PipelineState:
    llm = get_llm()
    vs_lessons = get_vectorstore(state["lesson_vector_table"])
    homework: Dict[str, str] = {}
    for s in state["students"]:
        weak = [c for c, v in state["scores"][s].items() if v < 70]
        if not weak:
            homework[s] = "🎉 Great job! No targeted homework—consider enrichment tasks from lesson resources."
            continue
        ctx = []
        for c in weak:
            docs = vs_lessons.similarity_search(c, k=3)
            ctx.extend([d.page_content[:400] for d in docs])
        context = "\n---\n".join(ctx[:1200])
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = [SystemMessage(content=SYSTEM_HW),
                HumanMessage(content=USER_HW.format(student_name=s, weak_concepts=", ".join(weak), context=context))]
        hw = llm.invoke(msgs).content
        homework[s] = hw
    write_homework(get_engine(), homework)
    state["homework"] = homework
    return state

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", node_ingest)
    g.add_node("evaluate", node_evaluate)
    g.add_node("reports", node_reports)
    g.add_node("kg_groups", node_knowledge_graph_and_groups)
    g.add_node("lessons", node_lesson_plans)
    g.add_node("homework", node_homework)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "evaluate")
    g.add_edge("evaluate", "reports")
    g.add_edge("reports", "kg_groups")
    g.add_edge("kg_groups", "lessons")
    g.add_edge("lessons", "homework")
    g.add_edge("homework", END)
    return g.compile()
