from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()


def tidb_connection_string() -> str:
    user = os.getenv("TIDB_USER")
    pwd = os.getenv("TIDB_PASSWORD")
    host = os.getenv("TIDB_HOST")
    port = int(os.getenv("TIDB_PORT", "4000"))
    db = os.getenv("TIDB_DATABASE", "test")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?ssl_ca='cert.pem'&ssl_verify_cert=true&ssl_verify_identity=true"

def get_engine() -> Engine:
    return create_engine(tidb_connection_string(), pool_pre_ping=True)

def ensure_relational_schema(engine: Engine):
    """
    Create normalized tables for comprehension, knowledge graph, groups, lessons, homework.
    Vector tables are created by TiDBVectorStore via LangChain.
    """
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS students (
            student_name VARCHAR(255) PRIMARY KEY
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS concepts (
            concept VARCHAR(255) PRIMARY KEY
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS comprehension (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            student_name VARCHAR(255),
            concept VARCHAR(255),
            score FLOAT,
            pain_points TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_student (student_name),
            INDEX idx_concept (concept),
            FOREIGN KEY (student_name) REFERENCES students(student_name) ON DELETE CASCADE,
            FOREIGN KEY (concept) REFERENCES concepts(concept) ON DELETE CASCADE
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS student_concepts (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            student_name VARCHAR(255),
            concept VARCHAR(255),
            awareness_score FLOAT,
            INDEX idx_sc_student (student_name),
            INDEX idx_sc_concept (concept),
            FOREIGN KEY (student_name) REFERENCES students(student_name) ON DELETE CASCADE,
            FOREIGN KEY (concept) REFERENCES concepts(concept) ON DELETE CASCADE
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS study_groups (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            group_id INT,
            student_name VARCHAR(255),
            INDEX idx_group (group_id),
            INDEX idx_group_student (group_id, student_name)
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS lesson_plans (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            concept VARCHAR(255),
            plan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS homework_personalized (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            student_name VARCHAR(255),
            homework TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""))

def upsert_students_and_concepts(engine: Engine, student_names: List[str], concepts: List[str]):
    with engine.begin() as conn:
        for s in student_names:
            conn.execute(text("INSERT IGNORE INTO students(student_name) VALUES (:s)"), {"s": s})
        for c in concepts:
            conn.execute(text("INSERT IGNORE INTO concepts(concept) VALUES (:c)"), {"c": c})

def write_comprehension(engine: Engine, student_name: str, concept: str, score: float, pain_points: List[str]):
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT IGNORE INTO comprehension (student_name, concept, score, pain_points)
                    VALUES (:s, :c, :score, :pp)"""),
            {"s": student_name, "c": concept, "score": float(score), "pp": "\n".join(pain_points)}
        )

def write_student_concepts(engine: Engine, scores: Dict[str, Dict[str, float]]):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM student_concepts"))
        for s, m in scores.items():
            for c, v in m.items():
                conn.execute(
                    text("""INSERT INTO student_concepts (student_name, concept, awareness_score)
                            VALUES (:s, :c, :v)"""),
                    {"s": s, "c": c, "v": float(v)}
                )

def write_study_groups(engine: Engine, groups: Dict[int, List[str]]):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM study_groups"))
        for gid, members in groups.items():
            for s in members:
                conn.execute(text("INSERT INTO study_groups (group_id, student_name) VALUES (:g, :s)"),
                             {"g": gid, "s": s})

def write_lesson_plans(engine: Engine, plans: Dict[str, str]):
    with engine.begin() as conn:
        for c, plan in plans.items():
            conn.execute(text("INSERT INTO lesson_plans (concept, plan) VALUES (:c, :p)"),
                         {"c": c, "p": plan})

def write_homework(engine: Engine, hw: Dict[str, str]):
    with engine.begin() as conn:
        for s, text_hw in hw.items():
            conn.execute(text("INSERT INTO homework_personalized (student_name, homework) VALUES (:s, :h)"),
                         {"s": s, "h": text_hw})
