# AI Chat System with RAG and Custom System Prompts

A powerful Streamlit application that combines AI chat with Retrieval-Augmented Generation (RAG) capabilities, allowing you to:

- Upload and query your own documents (PDF, DOCX, TXT)
- Create and manage multiple system prompts
- Chat with an AI using the selected system prompt
- Get AI responses enhanced with context from your documents
- Save and load your custom prompts
- Use custom API endpoints and models

## 🆕 Latest Updates

- **Improved File Upload**: Form-based upload with explicit submit button prevents repetition issues
- **Duplicate Detection**: Files are hashed to prevent uploading the same document twice
- **Cleaner UI**: File upload in collapsible expander, improved sidebar organization
- **Simplified Configuration**: API settings now managed via environment variables only
- **Better Error Handling**: Fixed deprecation warnings and tokenizer parallelism issues
- **Enhanced Document Display**: Compact document list with truncated filenames

## Quick Start

```bash
# Clone and enter directory
git clone https://github.com/yourusername/automateAI.git && cd automateAI

# Install dependencies
uv sync  # or pip install -r requirements.txt

# Set up your OpenAI API key
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the app
streamlit run demo.py
```

## Features

- **RAG (Retrieval-Augmented Generation)**: Upload documents and get AI responses based on your content
- **Smart Document Management**: Support for PDF, DOCX, and TXT files with automatic processing and duplicate detection
- **Vector Storage**: Local Qdrant vector database for efficient document retrieval (in-memory, no external server needed)
- **Multiple System Prompts**: Create, edit, and delete custom system prompts
- **Clean Chat Interface**: Interactive chat UI with organized file upload section
- **Duplicate Prevention**: Intelligent file hashing prevents the same document from being uploaded multiple times
- **API Flexibility**: Works with OpenAI, Azure OpenAI, and other compatible APIs via environment variables
- **Persistence**: Save and load your prompts between sessions
- **Context Display**: See which document chunks were used to generate responses with relevance scores

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/automateAI.git
   cd automateAI
   ```

2. Install dependencies using uv (recommended) or pip:

   Using uv:

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install qdrant-client openai streamlit pypdf python-docx tiktoken fastembed python-dotenv
   ```

3. Set up environment variables:

   Copy the example template:

   ```bash
   cp env.example .env
   ```

   Edit the `.env` file with your credentials:

   ```bash
   # Required
   OPENAI_API_KEY=your_api_key_here
   
   # Optional
   OPENAI_API_BASE=https://your-endpoint.openai.azure.com
   OPENAI_MODEL=gpt-4o-2024-11-20
   ```

## Usage

1. Run the application:

   ```bash
   streamlit run demo.py
   ```

2. The app will automatically use your configured API credentials from the `.env` file

3. Upload documents (optional):
   - Click the "📎 Upload Document" expander above the chat
   - Select a PDF, DOCX, or TXT file
   - Click "Upload and Process" to add it to your knowledge base
   - Files are automatically chunked, embedded, and indexed
   - Duplicate files are automatically detected and prevented

4. Manage your documents:
   - View all uploaded documents in the sidebar
   - See chunk counts for each document
   - Delete individual documents with the 🗑️ button
   - Toggle RAG on/off in the sidebar

5. Use system prompts:
   - Select from pre-configured prompts or create your own
   - Prompts are managed in the sidebar

6. Start chatting:
   - Type your questions in the chat input
   - When RAG is enabled, the AI will use your documents for context
   - View source documents used in responses by expanding "📎 Context Sources"

## Features Guide

### RAG (Retrieval-Augmented Generation)

1. **Uploading Documents**:
   - Click the "📎 Upload Document" expander above the chat interface
   - Browse and select your file (PDF, DOCX, or TXT)
   - Click "Upload and Process" to submit
   - Form automatically clears after successful upload
   - Duplicate detection prevents the same file from being processed twice
   - Success notification appears briefly at the top of the page

2. **Document Processing**:
   - Text is extracted from uploaded files
   - Content is split into optimal chunks using tiktoken (500 tokens with 50 token overlap)
   - Each chunk is converted to a 384-dimensional embedding using FastEmbed (BAAI/bge-small-en-v1.5)
   - Vectors are stored in local Qdrant database (in-memory, no external server required)
   - File hash tracking ensures efficient duplicate prevention

3. **Querying Documents**:
   - Simply type your questions in the chat
   - The system automatically searches for the 5 most relevant chunks
   - Context is seamlessly integrated into the AI's response
   - Relevance scores show how well each chunk matches your query
   - Expandable "📎 Context Sources" section shows which documents were used

4. **Managing Documents**:
   - Sidebar displays all uploaded documents with chunk counts
   - Total document count shown at the top
   - Delete individual documents with the 🗑️ button
   - Toggle RAG on/off to switch between document-based and general chat
   - Documents persist for the session (in-memory storage)

### Managing System Prompts

1. **Creating New Prompts**:
   - Enter a name and content for your prompt
   - Click "Add Prompt" to create it
   - The prompt will be selected automatically

2. **Editing Prompts**:
   - Select a prompt from the dropdown
   - Edit its content in the text area
   - Changes are saved automatically

3. **Deleting Prompts**:
   - Select the prompt you want to delete
   - Click the "Delete" button
   - You can't delete the last remaining prompt

4. **Saving & Loading**:
   - Click "Save Prompts" to store all prompts to disk
   - Click "Load Prompts" to restore previously saved prompts

### API Configuration

The API is configured through environment variables in your `.env` file:

- **API Key** (`OPENAI_API_KEY`): Required for chat functionality
- **Base URL** (`OPENAI_API_BASE`): Optional, for custom endpoints like Azure OpenAI
- **Model** (`OPENAI_MODEL`): Defaults to gpt-4o-2024-11-20

No manual configuration needed in the UI - the app automatically uses your environment settings.

## Requirements

- Python 3.7+
- Streamlit >= 1.48.1
- OpenAI Python library >= 1.99.9
- Qdrant Client >= 1.15.1
- PyPDF >= 5.1.0
- python-docx >= 1.1.2
- tiktoken >= 0.8.0
- fastembed >= 0.4.2

## System Prompts

The application comes with three default system prompts:

- **Default**: A general helpful assistant
- **Coding Assistant**: Specialized in programming help
- **Creative Writer**: Focused on creative writing assistance

You can customize these or add your own prompts through the sidebar interface.

## Advanced Configuration

### Environment Variables

The app supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | None |
| `OPENAI_API_BASE` | Custom API base URL (e.g., Azure) | None |
| `OPENAI_MODEL` | Default model to use | gpt-4o-2024-11-20 |

You can set these in a `.env` file or in your system environment.

### Configuration Options

- **Custom API Endpoints**: Set a custom base URL for Azure OpenAI or other providers
- **Custom Models**: Enter specific model names or deployment IDs not in the standard list
- **Model Auto-detection**: The app will try to detect available models for your API key
- **Intelligent Error Handling**: Provides helpful guidance for API and model access issues

## Development

- **Modifying the UI**: The entire application is contained in `demo.py`
- **Adding New Features**: The codebase is designed to be easily extendable
- **Custom Storage**: Prompts are stored in `system_prompts.json` by default

## Troubleshooting

### Common Issues

1. **Tokenizer Parallelism Warning**:
   - The app automatically sets `TOKENIZERS_PARALLELISM=false` to prevent warnings
   - This is normal and doesn't affect functionality

2. **File Upload Issues**:
   - Make sure to click "Upload and Process" after selecting a file
   - The form will clear automatically after successful upload
   - Check file size - very large files may take time to process

3. **Duplicate File Detection**:
   - The system uses file hashing to prevent duplicates
   - If you need to re-upload a modified file, make sure the content has changed

4. **Memory Usage**:
   - Qdrant runs in-memory, so uploaded documents are lost when the app restarts
   - For persistent storage, consider using Qdrant's disk-based or server modes

## Notes

- Your API key is required to use the chat functionality
- The application uses gpt-4o-2024-11-20 by default
- System prompts are saved locally in `system_prompts.json`
- Uploaded documents are stored in memory during the session
- The app is optimized for local development and personal use
