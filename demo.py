import streamlit as st
import openai
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import tiktoken
from typing import List, Dict, Any
import uuid
import hashlib

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Qdrant and embedding imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# File processing imports
from pypdf import PdfReader
from docx import Document

# Load environment variables from .env file if present
load_dotenv()

# Configuration
st.set_page_config(page_title="AI Chat System with RAG", layout="wide")

# Initialize Qdrant client (in-memory for local development)
@st.cache_resource
def init_qdrant():
    """Initialize Qdrant client with in-memory storage"""
    client = QdrantClient(":memory:")
    
    # Create collection if it doesn't exist
    collection_name = "documents"
    
    # Using fastembed for embeddings (384 dimensions for BAAI/bge-small-en-v1.5)
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    
    return client, collection_name

# Initialize embedding model
@st.cache_resource
def init_embedding_model():
    """Initialize the embedding model"""
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Get clients
qdrant_client, COLLECTION_NAME = init_qdrant()
embedding_model = init_embedding_model()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompts" not in st.session_state:
    st.session_state.system_prompts = {
        "Default": "You are a helpful AI assistant.",
        "Coding Assistant": "You are a coding assistant who helps with programming problems.",
        "Creative Writer": "You are a creative writing assistant who helps craft stories and narratives."
    }

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = "Default"

if "api_key" not in st.session_state:
    # Try to get API key from environment variable
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
    
if "model" not in st.session_state:
    # Try to get model from environment variable, default to gpt-4o
    st.session_state.model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
    
if "api_base" not in st.session_state:
    # Try to get API base URL from environment variable
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

# File processing functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        return str(file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def process_uploaded_file(uploaded_file) -> str:
    """Process uploaded file and extract text"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks using fastembed"""
    embeddings = list(embedding_model.embed(texts))
    return embeddings

def store_in_qdrant(chunks: List[str], embeddings: List[List[float]], filename: str, file_id: str):
    """Store embeddings in Qdrant"""
    points = []
    
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{file_id}_{idx}".encode()).hexdigest()
        
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "file_id": file_id,
                    "chunk_index": idx
                }
            )
        )
    
    # Upload points to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    return len(points)

def search_similar_chunks(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for similar chunks in Qdrant"""
    # Generate embedding for query
    query_embedding = list(embedding_model.embed([query]))[0]
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )
    
    # Extract relevant information
    results = []
    for result in search_results:
        results.append({
            "text": result.payload["text"],
            "filename": result.payload["filename"],
            "score": result.score
        })
    
    return results

def delete_file_from_qdrant(file_id: str):
    """Delete all chunks associated with a file from Qdrant"""
    # Get all points for this file
    all_points = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "file_id",
                    "match": {
                        "value": file_id
                    }
                }
            ]
        },
        limit=1000
    )[0]
    
    # Extract point IDs
    point_ids = [point.id for point in all_points]
    
    if point_ids:
        # Delete points
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=point_ids
        )
    
    return len(point_ids)

# Save and load functions for system prompts
def save_prompts():
    try:
        with open("system_prompts.json", "w") as f:
            json.dump(st.session_state.system_prompts, f, indent=2)
        return True, "Prompts saved successfully!"
    except Exception as e:
        return False, f"Error saving prompts: {str(e)}"

def load_prompts():
    try:
        if os.path.exists("system_prompts.json"):
            with open("system_prompts.json", "r") as f:
                loaded_prompts = json.load(f)
                
            # Validate the loaded data
            if not isinstance(loaded_prompts, dict) or not loaded_prompts:
                return False, "Invalid format in saved prompts file."
                
            # Always ensure Default prompt exists
            if "Default" not in loaded_prompts:
                loaded_prompts["Default"] = "You are a helpful AI assistant."
                
            # Keep other default prompts as backup if not in loaded prompts
            default_prompts = {
                "Coding Assistant": "You are a coding assistant who helps with programming problems.",
                "Creative Writer": "You are a creative writing assistant who helps craft stories and narratives."
            }
            
            # Merge with other defaults (keeping loaded ones when there's overlap)
            for key, value in default_prompts.items():
                if key not in loaded_prompts:
                    loaded_prompts[key] = value
                    
            # Update the session state
            st.session_state.system_prompts = loaded_prompts
            
            # Make sure selected prompt exists in loaded prompts
            if st.session_state.selected_prompt not in loaded_prompts:
                st.session_state.selected_prompt = next(iter(loaded_prompts))
                
            return True, f"Loaded {len(loaded_prompts)} prompts."
        else:
            return False, "No saved prompts file found."
    except Exception as e:
        return False, f"Error loading prompts: {str(e)}"

# Try to load saved prompts at startup
success, message = load_prompts()
if not success:
    print(f"Note: {message}")  # Just log this, don't show to user on startup

# Sidebar for settings
with st.sidebar:
    st.title("System Prompts")
    
    # RAG Settings
    st.markdown("### RAG Settings")
    st.session_state.use_rag = st.checkbox("Use RAG for responses", value=st.session_state.use_rag)
    
    # Uploaded Files Management
    st.markdown("### Uploaded Documents")
    if st.session_state.uploaded_files_metadata:
        st.caption(f"Total documents: {len(st.session_state.uploaded_files_metadata)}")
        for file_meta in st.session_state.uploaded_files_metadata:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    filename_display = file_meta['filename']
                    if len(filename_display) > 30:
                        filename_display = filename_display[:30] + "..."
                    st.text(f"📄 {filename_display}")
                    st.caption(f"{file_meta['chunks']} chunks")
                with col2:
                    if st.button("🗑️", key=f"delete_{file_meta['file_id']}", help="Delete document"):
                        # Delete from Qdrant
                        delete_file_from_qdrant(file_meta['file_id'])
                        # Remove from metadata
                        st.session_state.uploaded_files_metadata = [
                            f for f in st.session_state.uploaded_files_metadata 
                            if f['file_id'] != file_meta['file_id']
                        ]
                        st.rerun()
    else:
        st.info("No documents uploaded yet")
    
    st.divider()
    
    # Select prompt
    prompt_options = list(st.session_state.system_prompts.keys())
    
    # Handle case where selected prompt doesn't exist - reset to first prompt
    if st.session_state.selected_prompt not in prompt_options:
        if prompt_options:  # If we have any prompts
            st.session_state.selected_prompt = prompt_options[0]
            st.warning(f"The previously selected prompt '{st.session_state.selected_prompt}' was not found. Switched to '{prompt_options[0]}'.")
        else:
            # If no prompts at all (shouldn't happen with defaults)
            st.session_state.system_prompts = {
                "Default": "You are a helpful AI assistant."
            }
            st.session_state.selected_prompt = "Default"
            prompt_options = ["Default"]
            st.error("No system prompts found. Reset to default.")
    
    # Now safely get the index
    default_index = prompt_options.index(st.session_state.selected_prompt) if st.session_state.selected_prompt in prompt_options else 0
    
    # Define callback for dropdown change
    def on_prompt_change():
        if "selected_prompt_temp" in st.session_state:
            st.session_state.selected_prompt = st.session_state.selected_prompt_temp
            st.session_state.messages = []  # Clear chat when prompt changes
    
    # Use key and on_change parameters to handle selection with a single click
    selected_prompt = st.selectbox(
        "Choose a system prompt",
        options=prompt_options,
        index=default_index,
        key="selected_prompt_temp",
        on_change=on_prompt_change
    )
    
    # Display and edit the current system prompt
    st.subheader("Edit Current Prompt")
    current_prompt = st.text_area(
        "System prompt content", 
        value=st.session_state.system_prompts[selected_prompt],
        height=150
    )
    
    if current_prompt != st.session_state.system_prompts[selected_prompt]:
        st.session_state.system_prompts[selected_prompt] = current_prompt
    
    # Add new prompt
    st.subheader("Add/Update Prompt")
    
    # Store button click state to persist across reruns
    if "add_prompt_clicked" not in st.session_state:
        st.session_state.add_prompt_clicked = False
    
    if "overwrite_mode" not in st.session_state:
        st.session_state.overwrite_mode = False
        st.session_state.prompt_to_overwrite = ""
    
    # Show success message if prompt was just added
    if st.session_state.add_prompt_clicked:
        if st.session_state.overwrite_mode:
            st.success(f"Updated existing prompt: '{st.session_state.selected_prompt}'")
            st.session_state.overwrite_mode = False
            st.session_state.prompt_to_overwrite = ""
        else:
            st.success(f"Added new prompt: '{st.session_state.selected_prompt}'")
        # Reset flag
        st.session_state.add_prompt_clicked = False
    
    new_prompt_name = st.text_input("Prompt name")
    new_prompt_content = st.text_area("Prompt content", height=100)
    
    # Create columns for the buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        add_button = st.button("Add/Update Prompt")
    
    # If in overwrite confirmation mode, show confirmation button in second column
    if st.session_state.overwrite_mode and st.session_state.prompt_to_overwrite == new_prompt_name:
        with col2:
            confirm_button = st.button("Confirm Overwrite", type="primary")
            if confirm_button:
                # Overwrite the existing prompt
                st.session_state.system_prompts[new_prompt_name] = new_prompt_content
                st.session_state.selected_prompt = new_prompt_name
                st.session_state.messages = []  # Clear chat when prompt changes
                
                # Set flags
                st.session_state.add_prompt_clicked = True
                st.session_state.overwrite_mode = True
                
                # Save prompts automatically
                success, message = save_prompts()
                if not success:
                    st.error(f"Failed to save prompts: {message}")
                    
                # Force page rerun to update dropdown
                st.rerun()
    
    if add_button and new_prompt_name:
        # Check if prompt name already exists
        if new_prompt_name in st.session_state.system_prompts:
            # Set the overwrite mode flag and store the prompt name
            st.session_state.overwrite_mode = True
            st.session_state.prompt_to_overwrite = new_prompt_name
            st.warning(f"A prompt named '{new_prompt_name}' already exists. Click 'Confirm Overwrite' to update it or choose a different name.")
        else:
            # Add as a new prompt
            st.session_state.system_prompts[new_prompt_name] = new_prompt_content
            st.session_state.selected_prompt = new_prompt_name
            st.session_state.messages = []  # Clear chat when prompt changes
            
            # Set flag instead of showing success message immediately
            st.session_state.add_prompt_clicked = True
            
            # Save prompts automatically when adding a new one
            success, message = save_prompts()
            if not success:
                st.error(f"Failed to save prompts: {message}")
                
            # Force page rerun to update dropdown
            st.rerun()
    
    # Delete prompt
    st.subheader("Delete Current Prompt")
    
    # Track delete operation
    if "delete_clicked" not in st.session_state:
        st.session_state.delete_clicked = False
        st.session_state.deleted_prompt_name = ""
    
    # Show success message if a prompt was just deleted
    if st.session_state.delete_clicked:
        st.success(f"Deleted prompt: '{st.session_state.deleted_prompt_name}'")
        st.session_state.delete_clicked = False
        st.session_state.deleted_prompt_name = ""
    
    # Cannot delete the Default prompt
    if selected_prompt == "Default":
        st.info("The Default prompt cannot be deleted.")
        
    # Check if we have more than one prompt
    elif len(st.session_state.system_prompts) <= 1:
        st.info("You must have at least one prompt.")
        
    # Normal delete button
    elif st.button("Delete Current Prompt"):
        # Store name for success message
        prompt_to_delete = selected_prompt
        
        # Delete the prompt
        del st.session_state.system_prompts[prompt_to_delete]
        
        # Ensure the Default prompt exists
        if "Default" not in st.session_state.system_prompts:
            st.session_state.system_prompts["Default"] = "You are a helpful AI assistant."
            
        # Always set back to Default prompt
        st.session_state.selected_prompt = "Default"
        st.session_state.messages = []  # Clear chat when prompt changes
        
        # Save prompts automatically after deletion
        success, message = save_prompts()
        if not success:
            st.error(f"Failed to save prompts: {message}")
        
        # Set flag for success message
        st.session_state.delete_clicked = True
        st.session_state.deleted_prompt_name = prompt_to_delete
        
        # Force page rerun to update dropdown
        st.rerun()
    
    # Save and Load buttons
    st.subheader("Save/Load Prompts")
    
    # Track states for save/load operations
    if "save_clicked" not in st.session_state:
        st.session_state.save_clicked = False
        st.session_state.save_message = ""
        st.session_state.save_success = False
        
    if "load_clicked" not in st.session_state:
        st.session_state.load_clicked = False
        st.session_state.load_message = ""
        st.session_state.load_success = False
    
    # Display feedback from previous operations
    if st.session_state.save_clicked:
        if st.session_state.save_success:
            st.success(st.session_state.save_message)
        else:
            st.error(st.session_state.save_message)
        st.session_state.save_clicked = False
        
    if st.session_state.load_clicked:
        if st.session_state.load_success:
            st.success(st.session_state.load_message)
        else:
            st.warning(st.session_state.load_message)
        st.session_state.load_clicked = False
    
    # Save/Load buttons
    save_col, load_col = st.columns(2)
    with save_col:
        if st.button("Save Prompts"):
            success, message = save_prompts()
            st.session_state.save_clicked = True
            st.session_state.save_success = success
            st.session_state.save_message = message
            st.rerun()
    
    with load_col:
        if st.button("Load Prompts"):
            success, message = load_prompts()
            st.session_state.load_clicked = True
            st.session_state.load_success = success
            st.session_state.load_message = message
            st.rerun()
    
    # Reset chat button
    st.subheader("Chat Controls")
    if st.button("Reset Chat"):
        st.session_state.messages = []

# Main chat interface
st.title("AI Chat System with RAG")
st.subheader(f"Active System Prompt: {st.session_state.selected_prompt}")

# Display RAG status
if st.session_state.use_rag and st.session_state.uploaded_files_metadata:
    st.info(f"📚 RAG enabled with {len(st.session_state.uploaded_files_metadata)} document(s)")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Show context sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander("📎 Context Sources"):
                for source in msg["sources"]:
                    st.write(f"**{source['filename']}** (relevance: {source['score']:.2f})")
                    st.text(source['text'][:200] + "...")

# Function to generate AI responses with RAG
def generate_response(messages):
    if not st.session_state.api_key:
        return "⚠️ Please enter your OpenAI API key in the sidebar. You can get an API key from https://platform.openai.com/api-keys", []
    
    try:
        # Create client with optional base URL
        if st.session_state.api_base:
            client = openai.OpenAI(
                api_key=st.session_state.api_key,
                base_url=st.session_state.api_base
            )
        else:
            client = openai.OpenAI(api_key=st.session_state.api_key)
        
        # Get the last user message
        last_user_message = messages[-1]["content"] if messages else ""
        
        # Search for relevant context if RAG is enabled
        context_sources = []
        context_text = ""
        
        if st.session_state.use_rag and st.session_state.uploaded_files_metadata and last_user_message:
            # Search for similar chunks
            similar_chunks = search_similar_chunks(last_user_message, limit=5)
            
            if similar_chunks:
                context_sources = similar_chunks
                context_text = "\n\n".join([chunk["text"] for chunk in similar_chunks])
                
                # Augment the system prompt with context
                augmented_system_prompt = f"""{st.session_state.system_prompts[st.session_state.selected_prompt]}

You have access to the following context from uploaded documents. Use this information to answer questions when relevant:

{context_text}

Important: If the answer can be found in the context, use it. If not, you can use your general knowledge but mention that the information is not from the uploaded documents."""
            else:
                augmented_system_prompt = st.session_state.system_prompts[st.session_state.selected_prompt]
        else:
            augmented_system_prompt = st.session_state.system_prompts[st.session_state.selected_prompt]
        
        # Add system prompt
        full_messages = [
            {"role": "system", "content": augmented_system_prompt}
        ] + messages
        
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=full_messages,
            temperature=0.7,
        )
        
        return response.choices[0].message.content, context_sources
        
    except openai.AuthenticationError:
        return "❌ **Authentication Error**: The API key you provided is invalid or has expired. Please check your API key and try again.", []
    except openai.RateLimitError:
        return "⚠️ **Rate Limit Error**: You've exceeded your current quota with OpenAI. Check your plan and billing details at https://platform.openai.com/account/billing", []
    except openai.APIError as e:
        if "model_not_found" in str(e) or "does not have access to model" in str(e):
            return f"❌ **Model Access Error**: Your account doesn't have access to {st.session_state.model}. Try using gpt-3.5-turbo instead.", []
        return f"❌ **API Error**: {str(e)}", []
    except Exception as e:
        return f"❌ **Error**: {str(e)}", []

# Show upload success message if needed
if st.session_state.show_upload_success:
    st.success(st.session_state.upload_success_message)
    st.session_state.show_upload_success = False
    st.session_state.upload_success_message = ""

# File upload section
with st.expander("📎 Upload Document", expanded=False):
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, DOCX, or TXT files to add to your knowledge base"
        )
        
        submit_button = st.form_submit_button("Upload and Process")
        
        if submit_button and uploaded_file is not None:
            # Create a unique identifier for this file
            file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
            uploaded_file.seek(0)  # Reset file pointer after reading
            
            # Check if file has already been processed
            if file_hash not in st.session_state.processed_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Process the file
                    text = process_uploaded_file(uploaded_file)
                    
                    if text:
                        # Chunk the text
                        chunks = chunk_text(text)
                        
                        if chunks:
                            # Generate embeddings
                            embeddings = generate_embeddings(chunks)
                            
                            # Generate unique file ID
                            file_id = str(uuid.uuid4())
                            
                            # Store in Qdrant
                            num_chunks = store_in_qdrant(chunks, embeddings, uploaded_file.name, file_id)
                            
                            # Add to metadata
                            st.session_state.uploaded_files_metadata.append({
                                "filename": uploaded_file.name,
                                "file_id": file_id,
                                "chunks": num_chunks,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Mark file as processed
                            st.session_state.processed_files.add(file_hash)
                            
                            # Set success message
                            st.session_state.show_upload_success = True
                            st.session_state.upload_success_message = f"✅ Successfully uploaded {uploaded_file.name} ({num_chunks} chunks)"
                            
                            st.rerun()
                        else:
                            st.error("No content could be extracted from the file")
                    else:
                        st.error("Failed to process the file")
            else:
                st.warning(f"File '{uploaded_file.name}' has already been uploaded")

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = generate_response(st.session_state.messages)
            st.write(response)
            
            # Show sources if available
            if sources:
                with st.expander("📎 Context Sources"):
                    for source in sources:
                        st.write(f"**{source['filename']}** (relevance: {source['score']:.2f})")
                        st.text(source['text'][:200] + "...")
    
    # Add assistant response to chat history with sources
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })

# Display model and timestamp info
st.caption(f"Using system prompt: {st.session_state.selected_prompt}")
st.caption(f"Model: {st.session_state.model}")
st.caption(f"Session time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.session_state.uploaded_files_metadata:
    st.caption(f"Documents loaded: {len(st.session_state.uploaded_files_metadata)}")