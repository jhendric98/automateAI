from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Sequence

import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import APIError, AuthenticationError, OpenAI, RateLimitError
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from pypdf import PdfReader
from docx import Document

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------
PAGE_TITLE = "AI Chat System with RAG"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
PROMPT_FILE = "system_prompts.json"
DEFAULT_PROMPTS: Dict[str, str] = {
    "Default": "You are a helpful AI assistant.",
    "Coding Assistant": "You are a coding assistant who helps with programming problems.",
    "Creative Writer": "You are a creative writing assistant who helps craft stories and narratives.",
}
MAX_SCROLL_LIMIT = 1_000
MAX_CONTEXT_RESULTS = 10
DEFAULT_RETRIEVAL_LIMIT = 5
DEFAULT_TEMPERATURE = 0.7
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 50


@dataclass
class DocumentMetadata:
    filename: str
    file_id: str
    file_hash: str
    chunks: int
    timestamp: str

    def short_name(self, max_length: int = 30) -> str:
        """Return a truncated filename for UI display."""
        if len(self.filename) <= max_length:
            return self.filename
        return f"{self.filename[: max_length - 3]}..."

    def to_session(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_session(data: Dict[str, Any]) -> "DocumentMetadata":
        return DocumentMetadata(
            filename=data["filename"],
            file_id=data["file_id"],
            file_hash=data.get("file_hash", ""),
            chunks=data.get("chunks", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat(timespec="seconds")),
        )


@st.cache_resource
def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client with in-memory storage."""
    client = QdrantClient(":memory:")
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )
    return client


@st.cache_resource
def init_embedding_model() -> TextEmbedding:
    """Initialize the embedding model."""
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_encoder() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


qdrant_client = init_qdrant()
embedding_model = init_embedding_model()

st.set_page_config(page_title=PAGE_TITLE, layout="wide")


def init_session_state() -> None:
    """Ensure Streamlit session state is populated with defaults."""
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []

    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = DEFAULT_PROMPTS.copy()
    else:
        for name, prompt in DEFAULT_PROMPTS.items():
            st.session_state.system_prompts.setdefault(name, prompt)

    if "selected_prompt" not in st.session_state or (
        st.session_state.selected_prompt not in st.session_state.system_prompts
    ):
        st.session_state.selected_prompt = "Default"

    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

    if "model" not in st.session_state:
        st.session_state.model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    if "api_base" not in st.session_state:
        base_url = os.getenv("OPENAI_API_BASE", "")
        st.session_state.api_base = base_url

    if "uploaded_files_metadata" not in st.session_state:
        st.session_state.uploaded_files_metadata: List[Dict[str, Any]] = []

    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "show_upload_success" not in st.session_state:
        st.session_state.show_upload_success = False
        st.session_state.upload_success_message = ""

    if "retrieval_limit" not in st.session_state:
        st.session_state.retrieval_limit = DEFAULT_RETRIEVAL_LIMIT

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

    if "prompt_load_message" not in st.session_state:
        st.session_state.prompt_load_message = ""

    if "prompts_initialized" not in st.session_state:
        st.session_state.prompts_initialized = False


init_session_state()


def save_prompts() -> tuple[bool, str]:
    """Persist system prompts to disk."""
    try:
        with open(PROMPT_FILE, "w", encoding="utf-8") as file:
            json.dump(st.session_state.system_prompts, file, indent=2)
        return True, "Prompts saved successfully!"
    except OSError as exc:
        return False, f"Error saving prompts: {exc}"


def load_prompts() -> tuple[bool, str]:
    """Load system prompts from disk and merge with defaults."""
    prompt_path = Path(PROMPT_FILE)
    if not prompt_path.exists():
        return False, "No saved prompts file found."

    try:
        with prompt_path.open("r", encoding="utf-8") as file:
            loaded_prompts = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Error loading prompts: {exc}"

    if not isinstance(loaded_prompts, dict) or not loaded_prompts:
        return False, "Invalid format in saved prompts file."

    for name, prompt in DEFAULT_PROMPTS.items():
        loaded_prompts.setdefault(name, prompt)

    st.session_state.system_prompts = loaded_prompts

    if st.session_state.selected_prompt not in loaded_prompts:
        st.session_state.selected_prompt = next(iter(loaded_prompts))

    return True, f"Loaded {len(loaded_prompts)} prompts."


if not st.session_state.prompts_initialized:
    success, message = load_prompts()
    if not success:
        st.session_state.prompt_load_message = message
    st.session_state.prompts_initialized = True


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:  # pragma: no cover - depends on external library
        st.error(f"Error reading PDF: {exc}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text content from a DOCX file."""
    try:
        document = Document(BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(paragraphs)
    except Exception as exc:  # pragma: no cover - depends on external library
        st.error(f"Error reading DOCX: {exc}")
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text content from a plain text file."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


def process_uploaded_file(filename: str, file_bytes: bytes) -> str:
    """Process the uploaded file based on its extension and extract text."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == ".docx":
        return extract_text_from_docx(file_bytes)
    if extension == ".txt":
        return extract_text_from_txt(file_bytes)

    st.error(f"Unsupported file type: {extension}")
    return ""


def chunk_text(text: str, chunk_size: int = TEXT_CHUNK_SIZE, chunk_overlap: int = TEXT_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    chunk_size = max(chunk_size, chunk_overlap + 1)
    encoder = get_encoder()
    tokens = encoder.encode(text)
    chunks: List[str] = []

    for start in range(0, len(tokens), chunk_size - chunk_overlap):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def generate_embeddings(texts: Sequence[str]) -> List[List[float]]:
    """Generate embeddings for text chunks using fastembed."""
    if not texts:
        return []
    return list(embedding_model.embed(texts))


def store_in_qdrant(
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    filename: str,
    file_id: str,
) -> int:
    """Store embeddings in Qdrant and return the number of stored vectors."""
    points: List[PointStruct] = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=f"{file_id}-{index}",
                vector=list(embedding),
                payload={
                    "text": chunk,
                    "filename": filename,
                    "file_id": file_id,
                    "chunk_index": index,
                },
            )
        )

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def delete_file_from_qdrant(file_id: str) -> int:
    """Delete all chunks associated with a file from Qdrant."""
    scroll_filter = Filter(
        must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
    )

    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=MAX_SCROLL_LIMIT,
        with_payload=False,
    )

    point_ids = [point.id for point in points]
    if point_ids:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=point_ids),
        )
    return len(point_ids)


def search_similar_chunks(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search for similar chunks in Qdrant."""
    if not query.strip():
        return []

    try:
        query_embedding = next(embedding_model.embed([query]))
    except StopIteration:  # pragma: no cover - defensive
        return []

    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=list(query_embedding),
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - depends on Qdrant
        st.warning(f"Unable to search documents: {exc}")
        return []

    results: List[Dict[str, Any]] = []
    for result in search_results:
        payload = result.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "filename": payload.get("filename", "Unknown document"),
                "score": result.score,
            }
        )

    return results


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using the current configuration."""
    kwargs: Dict[str, Any] = {"api_key": st.session_state.api_key}
    api_base = st.session_state.api_base.strip()
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def render_context_sources(sources: Sequence[Dict[str, Any]]) -> None:
    """Render retrieved context sources in an expander."""
    if not sources:
        return

    with st.expander("📎 Context Sources"):
        for source in sources:
            filename = source.get("filename", "Unknown document")
            score = source.get("score", 0.0)
            text = source.get("text", "")
            st.markdown(f"**{filename}** (relevance: {score:.2f})")
            preview = text[:200] + ("..." if len(text) > 200 else "")
            st.text(preview)


def generate_response(messages: Sequence[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Generate an AI response optionally augmented with document context."""
    if not st.session_state.api_key:
        warning = (
            "⚠️ Please enter your OpenAI API key in the sidebar. You can get an API key from "
            "https://platform.openai.com/api-keys"
        )
        return warning, []

    try:
        client = get_openai_client()
    except Exception as exc:  # pragma: no cover - protective
        return f"❌ **Configuration Error**: {exc}", []

    last_user_message = next(
        (message["content"] for message in reversed(messages) if message.get("role") == "user"),
        "",
    )

    base_prompt = st.session_state.system_prompts.get(
        st.session_state.selected_prompt,
        DEFAULT_PROMPTS["Default"],
    )

    context_sources: List[Dict[str, Any]] = []
    augmented_prompt = base_prompt

    if (
        st.session_state.use_rag
        and st.session_state.uploaded_files_metadata
        and last_user_message
    ):
        similar_chunks = search_similar_chunks(last_user_message, st.session_state.retrieval_limit)
        if similar_chunks:
            context_sources = similar_chunks
            context_sections = [
                f"Source: {chunk['filename']}\n{chunk['text']}" for chunk in similar_chunks
            ]
            context_text = "\n\n---\n\n".join(context_sections)
            augmented_prompt = (
                f"{base_prompt}\n\n"
                "You have access to the following context from uploaded documents. "
                "Use it when it is relevant to the user's request:\n\n"
                f"{context_text}\n\n"
                "If the answer is not contained in the provided context, you may use your general knowledge "
                "but mention that the information is not sourced from the uploaded documents."
            )

    chat_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in {"user", "assistant"}
    ]

    chat_messages.insert(0, {"role": "system", "content": augmented_prompt})

    try:
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=chat_messages,
            temperature=float(st.session_state.temperature),
        )
        return response.choices[0].message.content, context_sources
    except AuthenticationError:
        return (
            "❌ **Authentication Error**: The API key you provided is invalid or has expired. "
            "Please check your API key and try again.",
            [],
        )
    except RateLimitError:
        return (
            "⚠️ **Rate Limit Error**: You've exceeded your current quota with OpenAI. "
            "Check your plan and billing details at https://platform.openai.com/account/billing.",
            [],
        )
    except APIError as exc:
        message = str(exc)
        if "model_not_found" in message or "does not have access to model" in message:
            return (
                f"❌ **Model Access Error**: Your account doesn't have access to {st.session_state.model}. "
                "Try using gpt-3.5-turbo instead.",
                [],
            )
        return f"❌ **API Error**: {message}", []
    except Exception as exc:  # pragma: no cover - protective
        return f"❌ **Error**: {exc}", []


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("System Prompts")

    if st.session_state.prompt_load_message:
        st.info(st.session_state.prompt_load_message)

    st.markdown("### Retrieval Settings")
    st.session_state.use_rag = st.checkbox(
        "Use RAG for responses", value=st.session_state.use_rag
    )
    st.session_state.retrieval_limit = st.slider(
        "Context chunks per query",
        min_value=1,
        max_value=MAX_CONTEXT_RESULTS,
        value=int(st.session_state.retrieval_limit),
    )
    st.session_state.temperature = st.slider(
        "Response temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.temperature),
        step=0.05,
    )

    st.markdown("### OpenAI Settings")
    st.session_state.api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="Stored locally in your browser session",
    )
    st.session_state.model = st.text_input(
        "Model",
        value=st.session_state.model,
        help="Example: gpt-4o-2024-11-20",
    )
    st.session_state.api_base = st.text_input(
        "API Base URL (optional)",
        value=st.session_state.api_base,
        help="Set when using Azure OpenAI or custom endpoints",
    )

    st.markdown("### Uploaded Documents")
    if st.session_state.uploaded_files_metadata:
        st.caption(f"Total documents: {len(st.session_state.uploaded_files_metadata)}")
        for file_meta_dict in st.session_state.uploaded_files_metadata:
            metadata = DocumentMetadata.from_session(file_meta_dict)
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📄 {metadata.short_name()}")
                    st.caption(f"{metadata.chunks} chunks • Uploaded {metadata.timestamp}")
                with col2:
                    if st.button(
                        "🗑️",
                        key=f"delete_{metadata.file_id}",
                        help="Delete document",
                    ):
                        delete_file_from_qdrant(metadata.file_id)
                        st.session_state.uploaded_files_metadata = [
                            item
                            for item in st.session_state.uploaded_files_metadata
                            if item["file_id"] != metadata.file_id
                        ]
                        st.session_state.processed_files.discard(metadata.file_hash)
                        st.toast(f"Deleted {metadata.filename}")
                        st.rerun()
    else:
        st.info("No documents uploaded yet")

    st.divider()

    prompt_options = list(st.session_state.system_prompts.keys())
    if st.session_state.selected_prompt not in prompt_options:
        st.session_state.selected_prompt = prompt_options[0]

    selected_index = prompt_options.index(st.session_state.selected_prompt)
    selected_prompt = st.selectbox(
        "Choose a system prompt",
        options=prompt_options,
        index=selected_index,
    )

    if selected_prompt != st.session_state.selected_prompt:
        st.session_state.selected_prompt = selected_prompt
        st.session_state.messages = []

    st.subheader("Edit Current Prompt")
    edited_prompt = st.text_area(
        "System prompt content",
        value=st.session_state.system_prompts[selected_prompt],
        height=150,
    )
    if edited_prompt != st.session_state.system_prompts[selected_prompt]:
        st.session_state.system_prompts[selected_prompt] = edited_prompt

    st.subheader("Add or Update Prompt")
    with st.form("add_prompt_form", clear_on_submit=True):
        new_prompt_name = st.text_input("Prompt name")
        new_prompt_content = st.text_area("Prompt content", height=120)
        overwrite_existing = new_prompt_name in st.session_state.system_prompts if new_prompt_name else False
        if overwrite_existing:
            st.info(f"A prompt named '{new_prompt_name}' already exists. Submitting will overwrite it.")
        submitted = st.form_submit_button("Save Prompt", type="primary")
        if submitted:
            if not new_prompt_name.strip():
                st.error("Prompt name cannot be empty.")
            elif not new_prompt_content.strip():
                st.error("Prompt content cannot be empty.")
            else:
                st.session_state.system_prompts[new_prompt_name] = new_prompt_content
                st.session_state.selected_prompt = new_prompt_name
                st.session_state.messages = []
                success, message = save_prompts()
                if success:
                    st.success(f"Prompt '{new_prompt_name}' saved.")
                else:
                    st.warning(message)
                st.rerun()

    st.subheader("Delete Current Prompt")
    if selected_prompt == "Default":
        st.info("The Default prompt cannot be deleted.")
    elif len(st.session_state.system_prompts) <= 1:
        st.info("You must have at least one prompt.")
    else:
        if st.button("Delete Current Prompt"):
            deleted_prompt = st.session_state.selected_prompt
            del st.session_state.system_prompts[deleted_prompt]
            if "Default" not in st.session_state.system_prompts:
                st.session_state.system_prompts["Default"] = DEFAULT_PROMPTS["Default"]
            st.session_state.selected_prompt = "Default"
            st.session_state.messages = []
            success, message = save_prompts()
            if success:
                st.success(f"Deleted prompt: '{deleted_prompt}'")
            else:
                st.warning(message)
            st.rerun()

    st.subheader("Save / Load Prompts")
    save_col, load_col = st.columns(2)
    with save_col:
        if st.button("Save Prompts"):
            success, message = save_prompts()
            if success:
                st.success(message)
            else:
                st.warning(message)
    with load_col:
        if st.button("Load Prompts"):
            success, message = load_prompts()
            if success:
                st.success(message)
            else:
                st.warning(message)
                st.session_state.prompt_load_message = message
            st.rerun()

    st.subheader("Chat Controls")
    if st.button("Reset Chat"):
        st.session_state.messages = []


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title(PAGE_TITLE)
st.subheader(f"Active System Prompt: {st.session_state.selected_prompt}")

if st.session_state.use_rag and st.session_state.uploaded_files_metadata:
    st.info(f"📚 RAG enabled with {len(st.session_state.uploaded_files_metadata)} document(s)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        render_context_sources(msg.get("sources", []))

if st.session_state.show_upload_success:
    st.success(st.session_state.upload_success_message)
    st.session_state.show_upload_success = False
    st.session_state.upload_success_message = ""

with st.expander("📎 Upload Document", expanded=False):
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "docx"],
            help="Upload PDF, DOCX, or TXT files to add to your knowledge base",
        )
        submit_button = st.form_submit_button("Upload and Process")

    if submit_button:
        if uploaded_file is None:
            st.warning("Please choose a file to upload.")
        else:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()

            if file_hash in st.session_state.processed_files:
                st.warning(f"File '{uploaded_file.name}' has already been uploaded.")
            else:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    text = process_uploaded_file(uploaded_file.name, file_bytes)

                    if not text.strip():
                        st.error("No content could be extracted from the file.")
                    else:
                        chunks = chunk_text(text)
                        if not chunks:
                            st.error("No content could be extracted from the file.")
                        else:
                            embeddings = generate_embeddings(chunks)
                            file_id = str(uuid.uuid4())
                            num_chunks = store_in_qdrant(chunks, embeddings, uploaded_file.name, file_id)

                            metadata = DocumentMetadata(
                                filename=uploaded_file.name,
                                file_id=file_id,
                                file_hash=file_hash,
                                chunks=num_chunks,
                                timestamp=datetime.now().isoformat(timespec="seconds"),
                            )
                            st.session_state.uploaded_files_metadata.append(metadata.to_session())
                            st.session_state.processed_files.add(file_hash)
                            st.session_state.show_upload_success = True
                            st.session_state.upload_success_message = (
                                f"✅ Successfully uploaded {uploaded_file.name} ({num_chunks} chunks)"
                            )
                            st.rerun()

prompt = st.chat_input("What would you like to ask?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = generate_response(st.session_state.messages)
            st.write(response)
            render_context_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

st.caption(f"Using system prompt: {st.session_state.selected_prompt}")
st.caption(f"Model: {st.session_state.model}")
st.caption(f"Session time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.session_state.uploaded_files_metadata:
    st.caption(f"Documents loaded: {len(st.session_state.uploaded_files_metadata)}")
