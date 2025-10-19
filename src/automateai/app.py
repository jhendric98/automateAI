from __future__ import annotations

from datetime import datetime

import os
import hashlib
import uuid
import streamlit as st  # type: ignore
from importlib import resources as importlib_resources

from .rag import (
    chunk_text,
    delete_file_from_qdrant,
    generate_embeddings,
    init_embedding_model,
    init_qdrant,
    process_uploaded_file,
    store_in_qdrant,
    search_similar_chunks,
)


def main() -> None:
    """Streamlit application entry point."""
    # Disable tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load env if available
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    st.set_page_config(page_title="AI Chat System with RAG", layout="wide")

    # Initialize services
    qdrant_client, COLLECTION_NAME = init_qdrant()
    embedding_model = init_embedding_model()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompts" not in st.session_state:
        # Load packaged defaults
        try:
            with importlib_resources.files("automateai.data").joinpath("system_prompts.json").open(
                "r", encoding="utf-8"
            ) as f:
                import json

                st.session_state.system_prompts = json.load(f)
        except Exception:
            st.session_state.system_prompts = {
                "Default": "You are a helpful AI assistant.",
                "Coding Assistant": "You are a coding assistant who helps with programming problems.",
                "Creative Writer": "You are a creative writing assistant who helps craft stories and narratives.",
            }
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = "Default"
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
    if "model" not in st.session_state:
        st.session_state.model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    if "api_base" not in st.session_state:
        base_url = os.getenv("OPENAI_API_BASE", "")
        st.session_state.api_base = base_url if base_url else None
    if "uploaded_files_metadata" not in st.session_state:
        st.session_state.uploaded_files_metadata = []
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "show_upload_success" not in st.session_state:
        st.session_state.show_upload_success = False
        st.session_state.upload_success_message = ""

    # Sidebar
    with st.sidebar:
        st.title("System Prompts")
        st.markdown("### RAG Settings")
        st.session_state.use_rag = st.checkbox(
            "Use RAG for responses", value=st.session_state.use_rag
        )
        st.markdown("### Uploaded Documents")
        if st.session_state.uploaded_files_metadata:
            st.caption(
                f"Total documents: {len(st.session_state.uploaded_files_metadata)}"
            )
            for file_meta in st.session_state.uploaded_files_metadata:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        filename_display = file_meta["filename"]
                        if len(filename_display) > 30:
                            filename_display = filename_display[:30] + "..."
                        st.text(f"📄 {filename_display}")
                        st.caption(f"{file_meta['chunks']} chunks")
                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{file_meta['file_id']}",
                            help="Delete document",
                        ):
                            delete_file_from_qdrant(
                                qdrant_client, COLLECTION_NAME, file_meta["file_id"]
                            )
                            st.session_state.uploaded_files_metadata = [
                                f
                                for f in st.session_state.uploaded_files_metadata
                                if f["file_id"] != file_meta["file_id"]
                            ]
                            st.rerun()
        else:
            st.info("No documents uploaded yet")

        st.divider()
        prompt_options = list(st.session_state.system_prompts.keys())
        if st.session_state.selected_prompt not in prompt_options:
            if prompt_options:
                st.session_state.selected_prompt = prompt_options[0]
            else:
                st.session_state.system_prompts = {
                    "Default": "You are a helpful AI assistant."
                }
                st.session_state.selected_prompt = "Default"
                prompt_options = ["Default"]

        def on_prompt_change() -> None:
            if "selected_prompt_temp" in st.session_state:
                st.session_state.selected_prompt = st.session_state.selected_prompt_temp
                st.session_state.messages = []

        selected_prompt = st.selectbox(
            "Choose a system prompt",
            options=prompt_options,
            index=prompt_options.index(st.session_state.selected_prompt),
            key="selected_prompt_temp",
            on_change=on_prompt_change,
        )

        st.subheader("Edit Current Prompt")
        current_prompt = st.text_area(
            "System prompt content",
            value=st.session_state.system_prompts[selected_prompt],
            height=150,
        )
        if current_prompt != st.session_state.system_prompts[selected_prompt]:
            st.session_state.system_prompts[selected_prompt] = current_prompt

        st.subheader("Save/Load Prompts")
        if st.button("Reset Chat"):
            st.session_state.messages = []

    st.title("AI Chat System with RAG")
    st.subheader(f"Active System Prompt: {st.session_state.selected_prompt}")
    if st.session_state.use_rag and st.session_state.uploaded_files_metadata:
        st.info(
            f"📚 RAG enabled with {len(st.session_state.uploaded_files_metadata)} document(s)"
        )

    # Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📎 Context Sources"):
                    for source in msg["sources"]:
                        st.write(
                            f"**{source['filename']}** (relevance: {source['score']:.2f})"
                        )
                        st.text(source["text"][:200] + "...")

    # Upload section
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
            if submit_button and uploaded_file is not None:
                file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
                uploaded_file.seek(0)
                if file_hash not in st.session_state.processed_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        text = process_uploaded_file(uploaded_file)
                        if text:
                            chunks = chunk_text(text)
                            if chunks:
                                embeddings = generate_embeddings(
                                    embedding_model, chunks
                                )
                                file_id = str(uuid.uuid4())
                                num_chunks = store_in_qdrant(
                                    qdrant_client,
                                    COLLECTION_NAME,
                                    chunks,
                                    embeddings,
                                    uploaded_file.name,
                                    file_id,
                                )
                                st.session_state.uploaded_files_metadata.append(
                                    {
                                        "filename": uploaded_file.name,
                                        "file_id": file_id,
                                        "chunks": num_chunks,
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                )
                                st.session_state.processed_files.add(file_hash)
                                st.session_state.show_upload_success = True
                                st.session_state.upload_success_message = (
                                    f"✅ Successfully uploaded {uploaded_file.name} ({num_chunks} chunks)"
                                )
                                st.rerun()
                            else:
                                st.error("No content could be extracted from the file")
                        else:
                            st.error("Failed to process the file")
                else:
                    st.warning(f"File '{uploaded_file.name}' has already been uploaded")

    # Chat input and response
    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Optional OpenAI dependency; keep UI responsive even if not configured
                context_sources: list[dict[str, object]] = []
                if st.session_state.use_rag and st.session_state.uploaded_files_metadata:
                    context_sources = search_similar_chunks(
                        qdrant_client, COLLECTION_NAME, embedding_model, prompt, limit=5
                    )
                response_text = (
                    "RAG context gathered. Connect OpenAI to generate responses."
                )
                st.write(response_text)
                if context_sources:
                    with st.expander("📎 Context Sources"):
                        for source in context_sources:
                            st.write(
                                f"**{source['filename']}** (relevance: {source['score']:.2f})"
                            )
                            st.text(source["text"][:200] + "...")
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "sources": context_sources}
        )
    st.caption(f"Using system prompt: {st.session_state.selected_prompt}")
    st.caption(f"Model: {st.session_state.model}")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":  # pragma: no cover
    main()


