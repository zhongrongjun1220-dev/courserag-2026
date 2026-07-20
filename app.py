"""Streamlit frontend for CourseRAG.

This module owns the UI, chat history persistence, document ingestion, vector
store queries, and answer generation. Configuration is centralized in
``AppConfig`` so behavior can be tuned from one place."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from openai import OpenAI
import streamlit as st

from loaders import build_documents_from_file


APP_ROOT = Path(__file__).resolve().parent
EXPORT_DIR = APP_ROOT / "exports"
PERSIST_DIR = APP_ROOT / "storage"
HISTORY_FILE = PERSIST_DIR / "chat_history.json"
COLLECTION_NAME = "course-rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-5.5"
TOP_K = 4
MAX_CONTEXT_CHARS = 6000

EXPORT_DIR.mkdir(exist_ok=True)
PERSIST_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    """Centralize tunable RAG and UI settings."""

    collection_name: str = COLLECTION_NAME
    embedding_model: str = EMBEDDING_MODEL
    model: str = OPENAI_MODEL
    top_k: int = TOP_K
    max_context_chars: int = MAX_CONTEXT_CHARS
    history_file: Path = HISTORY_FILE


CONFIG = AppConfig()
st.markdown(
    """
    <style>
    .main [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.4rem !important;
        line-height: 1.35 !important;
    }
    details > summary {
        font-size: 0.95rem !important;
        line-height: 1.35 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("### 📚 CourseRAG")
st.caption("上传课程资料，基于本地向量库做语义检索问答，并导出引用与结果。")


@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached local Chroma client."""
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


@st.cache_resource
def get_embedding_function() -> ONNXMiniLM_L6_V2:
    """Return a cached ONNX embedding function."""
    return ONNXMiniLM_L6_V2()


def get_collection() -> chromadb.Collection:
    """Get or create the shared course document collection."""
    chroma_client = get_chroma_client()
    embedding_function = get_embedding_function()
    return chroma_client.get_or_create_collection(
        name=CONFIG.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def get_openai_client() -> OpenAI:
    """Build an OpenAI-compatible client from environment configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("请先在环境变量中设置 OPENAI_API_KEY。")
    return OpenAI(api_key=api_key, base_url=base_url)


def ingest_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> tuple[int, List[dict]]:
    """Persist uploaded files into the local vector store."""
    collection = get_collection()
    total_chunks = 0
    sources = []
    for file in files:
        save_path = APP_ROOT / "data" / file.name
        save_path.parent.mkdir(exist_ok=True)
        save_path.write_bytes(file.getbuffer())
        documents = build_documents_from_file(save_path)
        texts = [document["text"] for document in documents]
        metadatas = [
            {
                "chunk_id": document.get("chunk_id") or f"chunk_{idx + 1:03d}",
                "source": file.name,
                "page": document.get("page", 1),
                "chunk_index": document.get("chunk_index", idx + 1),
                "section": document.get("section") or "文档正文",
                "file_path": str(save_path),
                "char_start": document.get("char_start", 0),
                "char_end": document.get("char_end", len(document["text"])),
                "text_preview": document["text"][:220],
            }
            for idx, document in enumerate(documents)
        ]
        if texts:
            collection.add(
                ids=[f"{file.name}::{idx}" for idx in range(len(texts))],
                documents=texts,
                metadatas=metadatas,
            )
            total_chunks += len(texts)
            sources.append({"file": file.name, "chunks": len(texts)})
    return total_chunks, sources


def query_vector_store(question: str) -> List[dict]:
    """Retrieve the most relevant chunks for a user question."""
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=CONFIG.top_k)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    source_documents = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        source_documents.append(
            {
                "page_content": text,
                "metadata": metadata or {},
                "relevance_score": 1 - distance if distance is not None else None,
            }
        )
    return source_documents


def build_context(source_documents: List[dict]) -> tuple[str, List[dict]]:
    """Assemble retrieved chunks into a compact prompt context."""
    if not source_documents:
        return "", []
    current = ""
    selected = []
    for document in source_documents:
        metadata = document.get("metadata") or {}
        section = metadata.get("section") or "文档正文"
        chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "?"
        snippet = (
            f"[来源文件: {metadata.get('source', 'unknown')} | 章节: {section} | Chunk: {chunk_id} | 页码: {metadata.get('page', 1)}]\n"
            f"{document.get('page_content', '')}"
        )
        if len(current) + len(snippet) + 1 > CONFIG.max_context_chars:
            break
        current += snippet + "\n"
        selected.append(document)
    return current.strip(), selected


def generate_answer(question: str, source_documents: List[dict]) -> tuple[str, List[dict]]:
    """Create a grounded answer from retrieved course chunks."""
    context, selected_documents = build_context(source_documents)
    user_prompt = (
        "请优先依据资料作答；若多个来源都提到同一结论，请直接总结；"
        "若资料未直接给出答案，请基于资料中明确可推断的结论回答，并说明推断依据，不要仅回答'根据提供资料无法确定'。\n\n"
        f"检索到的资料：\n{context}\n\n问题：{question}"
        if context
        else "请优先依据资料作答；若资料未直接给出答案，请基于资料中明确可推断的结论回答，并说明推断依据，不要仅回答'根据提供资料无法确定'。\n\n问题：{question}"
    )
    completion = get_openai_client().chat.completions.create(
        model=CONFIG.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是课程问答助手。回答必须尽可能使用给定资料；"
                    "如果检索到的资料不完全，仍可基于资料中明确可推断的结论作答；"
                    "只有在资料确实完全缺失时，才说无法回答。\n"
                    "请用中文回答，并保持简洁、条理化。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content, selected_documents


def render_sources(sources: List[dict]) -> None:
    """Render expandable source citations beneath each answer."""
    if not sources:
        return
    st.markdown("### 引用来源")
    for idx, source in enumerate(sources, start=1):
        metadata = source.get("metadata") or {}
        relevance = source.get("relevance_score")
        relevance_text = f"{relevance:.2f}" if isinstance(relevance, (int, float)) else "?"
        with st.expander(
            f"[{idx}] {metadata.get('source', 'unknown')} | 章节: {metadata.get('section', '文档正文')} | chunk_id: {metadata.get('chunk_id', 'chunk_001')} | 匹配度: {relevance_text}"
        ):
            preview = metadata.get("text_preview") or source.get("page_content") or ""
            st.write(preview + ("..." if len(preview) >= 180 else ""))


def export_result(question: str, answer: str, sources: List[dict]) -> None:
    """Expose single-turn export controls for Markdown and JSON."""
    payload = {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": source.get("metadata", {}).get("source", "unknown"),
                "page": source.get("metadata", {}).get("page", 1),
                "chunk": source.get("metadata", {}).get("chunk_index", idx + 1),
                "preview": source.get("page_content", "")[:220],
            }
            for idx, source in enumerate(sources)
        ],
        "chunk_size": CONFIG.top_k,
        "chunk_overlap": 60,
        "embedding_model": CONFIG.embedding_model,
    }
    md_lines = [
        "# CourseRAG 问答结果",
        "",
        "## 问题",
        "",
        question,
        "",
        "## 回答",
        "",
        answer,
        "",
        "## 引用来源",
        "",
    ]
    for idx, source in enumerate(payload["sources"], start=1):
        md_lines.append(f"{idx}. **{source['source']}** | 页码:{source['page']} | 片段{source['chunk']}")
        md_lines.append("")
        md_lines.append(f"> {source['preview']}")
        md_lines.append("")

    col1, col2 = st.columns(2)
    col1.download_button("导出 Markdown", "\n".join(md_lines), file_name="course-qa.md", mime="text/markdown")
    col2.download_button("导出 JSON", json.dumps(payload, ensure_ascii=False, indent=2), file_name="course-qa.json", mime="application/json")


def _load_persisted_history() -> List[dict]:
    """Load previously saved chat history from disk."""
    if not CONFIG.history_file.exists():
        return []
    try:
        payload = json.loads(CONFIG.history_file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict) and "role" in item and "content" in item]
    except Exception:
        pass
    return []


def _save_persisted_history(history: List[dict]) -> None:
    """Write the current chat history to disk."""
    CONFIG.history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _restore_history() -> List[dict]:
    """Restore persisted history into the current Streamlit session."""
    persisted = _load_persisted_history()
    if persisted:
        st.session_state.history = persisted
    elif "history" not in st.session_state:
        st.session_state.history = []


def _persist_history() -> None:
    """Persist the current session history."""
    _save_persisted_history(st.session_state.get("history", []))


def render_history_exporter() -> None:
    """Render sidebar controls for exporting the full chat history."""
    st.sidebar.markdown("### 历史记录")
    if "history" not in st.session_state or not st.session_state.history:
        st.sidebar.caption("当前还没有聊天历史，可以继续问答后在该处导出。")
        st.sidebar.download_button(
            "预览空历史 JSON",
            json.dumps({"title": "CourseRAG 全部历史", "turns": 0, "history": []}, ensure_ascii=False, indent=2),
            file_name="course-qa-history.json",
            mime="application/json",
            disabled=True,
        )
        return

    history = st.session_state.history
    md_lines = ["# CourseRAG 全部历史", "", f"共 {len([message for message in history if message['role'] == 'user'])} 轮问答", "", "---", ""]
    turn = 0
    user_message = None
    payload = []
    for message in history:
        if message["role"] == "user":
            user_message = message
        elif message["role"] == "assistant" and user_message is not None:
            turn += 1
            md_lines.extend([f"## 问答 {turn}", "", "### 用户问题", "", user_message["content"], "", "### 助手回答", "", message["content"], ""])
            payload.append({"turn": turn, "question": user_message["content"], "answer": message["content"]})
            user_message = None

    json_payload = {
        "title": "CourseRAG 全部历史",
        "turns": turn,
        "history": payload,
    }
    st.sidebar.download_button("导出全部历史 Markdown", "\n".join(md_lines), file_name="course-qa-history.md", mime="text/markdown")
    st.sidebar.download_button("导出全部历史 JSON", json.dumps(json_payload, ensure_ascii=False, indent=2), file_name="course-qa-history.json", mime="application/json")


def main() -> None:
    """Run the main CourseRAG chat interface."""
    st.sidebar.header("资料库")
    uploaded_files = st.sidebar.file_uploader(
        "上传课程资料",
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "csv", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        with st.spinner("正在解析并写入本地向量库："):
            total_chunks, sources = ingest_files(uploaded_files)
        st.sidebar.success(f"已导入 {len(uploaded_files)} 个文件，共 {total_chunks} 个片段")
        for source in sources:
            st.sidebar.write(f"- {source['file']}: {source['chunks']} 个片段")

    _restore_history()
    render_history_exporter()

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入课程相关问题"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索资料并生成回答："):
                try:
                    source_documents = query_vector_store(prompt)
                    answer, selected_documents = generate_answer(prompt, source_documents)
                    selected_documents = selected_documents[:CONFIG.top_k]
                except Exception as exception:
                    answer = f"运行失败：{exception}"
                    selected_documents = []
            st.markdown("### 回答")
            st.write(answer)
            render_sources(selected_documents)
            export_result(prompt, answer, selected_documents)
            _persist_history()
        st.session_state.history.append({"role": "assistant", "content": answer})
        _persist_history()


if __name__ == "__main__":
    main()
