from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_community.vectorstores import TiDBVectorStore
import os
from typing import TypedDict, List, Dict, Any, Optional
from .tidb import *
import uuid

load_dotenv()
os.environ["OPENAI_API_KEY"] =os.getenv('OPENAI_API_KEY')

def get_embeddings():
    return OpenAIEmbeddings()

def get_llm():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.2)

def get_vectorstore(table_name: str) -> TiDBVectorStore:
    """
    Returns (or creates) a TiDB vector table through LangChain.
    The table schema includes id, embedding (VECTOR), document, meta (JSON), timestamps.  # docs-backed
    """
    vs = TiDBVectorStore(
        connection_string=tidb_connection_string(),
        embedding_function=get_embeddings(),
        table_name=table_name,
        distance_strategy="cosine",
    )
    return vs

def load_pdfs_to_docs(file, student_name: str) -> List[Document]:
    # Save to tmp and load
    tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader(tmp_path)
    raw_docs = loader.load()
    # Attach metadata for student
    for d in raw_docs:
        d.metadata = d.metadata or {}
        d.metadata.update({
            "student_name": student_name,
            "source_file": file.name
        })
    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(raw_docs)
    return docs

def ingest_documents(docs: List[Document], vs: TiDBVectorStore):
    vs.add_documents(docs)

def retrieve_student_context(vs: TiDBVectorStore, student_name: str, concept: str, k: int = 6) -> List[Document]:
    """
    Try metadata filter (meta JSON) if supported; otherwise post-filter.
    """
    try:
        docs = vs.similarity_search(f"{concept}", k=16, filter={"student_name": student_name})
    except Exception: # Fallback just in case
        docs = vs.similarity_search(f"{concept}", k=32)
        docs = [d for d in docs if (d.metadata or {}).get("student_name") == student_name]
    return docs[:k]
