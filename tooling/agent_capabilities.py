
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import TypedDict, List, Dict, Any, Optional
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] =os.getenv('OPENAI_API_KEY')

SYSTEM_SCORER = """You are a strict but fair grader. You will read the student's submission snippets and assess their comprehension of a target concept.
Return concise JSON with keys: score (0-100), pain_points (array of short strings), evidence (array of short quotes).
Score rubric:
- 90-100: Mastery (precise, transferable, correct terminology)
- 70-89: Proficient (mostly correct, minor gaps)
- 50-69: Developing (partial understanding, notable gaps)
- 0-49: Beginning (confused, misconceptions or missing)
"""
USER_SCORER = """Target Concept: "{concept}"
Student: {student_name}
Relevant snippets (not verbatim full text, only selected chunks):
{snippets}
Instructions:
- Use only the snippets and general knowledge of the concept (avoid hallucinations).
- Output JSON ONLY, no markdown.
"""

def score_comprehension(llm: ChatOpenAI, student_name: str, concept: str, snippets: List[str]) -> Dict[str, Any]:
    from langchain_core.messages import SystemMessage, HumanMessage
    user = USER_SCORER.format(concept=concept, student_name=student_name, snippets="\n\n---\n\n".join(snippets))
    msgs = [SystemMessage(content=SYSTEM_SCORER), HumanMessage(content=user)]
    resp = llm.invoke(msgs).content
    try:
        data = json.loads(resp)
    except Exception:
        start = resp.find("{")
        end = resp.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(resp[start:end+1])
        else:
            data = {"score": 0, "pain_points": ["Unable to parse grader output"], "evidence": []}
    data.setdefault("score", 0)
    data.setdefault("pain_points", [])
    data.setdefault("evidence", [])
    return data

SYSTEM_REPORT = """You are a teaching assistant generating a brief, actionable student report based on comprehension scores per concept.
Return markdown structured with: Summary, Strengths (bullets), Pain Points (bullets), Recommended Next Steps (bullets)."""
USER_REPORT = """Student: {student_name}
Per-concept results as JSON:
{results_json}
"""

SYSTEM_LESSON = """You are a curriculum designer. Create a concise, high-impact lesson plan that improves comprehension gaps across the whole class.
Plan format (markdown): Goals, Prerequisites, Mini-lessons (15-20 min), Practice Activities, Formative Checks, Exit Ticket, Suggested Resources.
Be specific and pragmatic."""
USER_LESSON = """Cohort-wide weaknesses (concepts with low scores): {weak_concepts}
Ground your suggestions in the short context snippets (if any).
Context:
{context}
"""

SYSTEM_HW = """You are a teacher creating personalized homework focused on the student's weakest concepts. 
Return markdown with sections by concept: 1) Micro-recap (100-150 words), 2) 3-5 targeted problems, 3) One extension/thought question."""
USER_HW = """Student: {student_name}
Weak concepts: {weak_concepts}
Use the lesson context (snippets) to shape tasks:
{context}
"""

def build_study_groups(scores: Dict[str, Dict[str, float]], concepts: List[str], target_size: int = 2) -> Dict[int, List[str]]:
    """
    Greedy heuristic:
    - While unassigned students exist, build a group by covering all concepts:
      pick student covering most uncovered concepts, then add complementary students.
    - Fill remaining slots with students that maximize marginal coverage.
    """
    aware = {c: {s for s, m in scores.items() if m.get(c, 0) >= 70.0} for c in concepts}
    all_students = set(scores.keys())
    unassigned = set(all_students)
    groups: Dict[int, List[str]] = {}
    gid = 1
    def coverage(group: set) -> set:
        covered = set()
        for c in concepts:
            if len(aware[c].intersection(group)) > 0:
                covered.add(c)
        return covered
    while unassigned:
        group = set()
        covered = set()
        best = max(unassigned, key=lambda s: sum(1 for c in concepts if s in aware[c]), default=None)
        if best:
            group.add(best); unassigned.remove(best)
            covered = coverage(group)
        while (covered != set(concepts)) and len(group) < target_size and unassigned:
            pick = max(unassigned, key=lambda s: len(coverage(group | {s}) - covered))
            group.add(pick); unassigned.remove(pick)
            covered = coverage(group)
        while len(group) < target_size and unassigned:
            pick = max(unassigned, key=lambda s: len(coverage(group | {s}) - covered))
            group.add(pick); unassigned.remove(pick)
            covered = coverage(group)
        groups[gid] = sorted(group)
        gid += 1
    return groups

def plot_knowledge_graph(scores: Dict[str, Dict[str, float]], concepts: List[str], aware_threshold: float = 70.0) -> plt.Figure:
    G = nx.Graph()
    for s in scores:
        G.add_node(s, bipartite=0, color="#3b82f6")  # students
    for c in concepts:
        G.add_node(c, bipartite=1, color="#10b981")  # concepts
    for s, m in scores.items():
        for c, v in m.items():
            if v >= aware_threshold:
                G.add_edge(s, c, weight=1.0)
    pos = nx.spring_layout(G, seed=42, k=0.7)
    colors = [G.nodes[n].get("color", "#999999") for n in G.nodes()]
    fig = plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800, font_size=8, edge_color="#7e2222")
    return fig
