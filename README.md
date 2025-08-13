# AI Chat System with RAG and Custom System Prompts

A powerful Streamlit application that combines AI chat with Retrieval-Augmented Generation (RAG) capabilities, allowing you to:

- Upload and query your own documents (PDF, DOCX, TXT)
- Create and manage multiple system prompts
- Chat with an AI using the selected system prompt
- Get AI responses enhanced with context from your documents
- Save and load your custom prompts
- Use custom API endpoints and models

## Features

- **RAG (Retrieval-Augmented Generation)**: Upload documents and get AI responses based on your content
- **Document Management**: Support for PDF, DOCX, and TXT files with automatic processing
- **Vector Storage**: Local Qdrant vector database for efficient document retrieval
- **Multiple System Prompts**: Create, edit, and delete custom system prompts
- **Chat Interface**: Interactive chat UI similar to ChatGPT with file upload capability
- **Model Selection**: Choose from various AI models or enter custom model names
- **API Flexibility**: Works with OpenAI, Azure OpenAI, and other compatible APIs
- **Persistence**: Save and load your prompts between sessions
- **Context Display**: See which document chunks were used to generate responses

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/automateAI.git
   cd automateAI
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
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

2. Set up your API key via environment variable:
   - Edit the `.env` file with your OpenAI API key
   - The app will automatically use the configured credentials

3. Upload documents (optional):
   - Click the 📎 button next to the chat input
   - Select PDF, DOCX, or TXT files to upload
   - Documents are automatically processed and indexed

4. Use the pre-configured system prompts or create your own

5. Start chatting with the AI:
   - Ask questions about your uploaded documents
   - Or have regular conversations without documents
   - Toggle RAG on/off in the sidebar as needed

## Features Guide

### RAG (Retrieval-Augmented Generation)

1. **Uploading Documents**:
   - Click the 📎 button next to the chat input
   - Select PDF, DOCX, or TXT files
   - Files are automatically processed, chunked, and indexed
   - View uploaded documents in the sidebar

2. **Document Processing**:
   - Text is extracted from uploaded files
   - Content is split into optimal chunks using tiktoken
   - Embeddings are generated using FastEmbed (BAAI/bge-small-en-v1.5)
   - Vectors are stored in local Qdrant database (in-memory)

3. **Querying Documents**:
   - Simply ask questions in the chat
   - The system automatically searches for relevant context
   - AI responses are enhanced with document content
   - View source chunks used for each response

4. **Managing Documents**:
   - View all uploaded documents in the sidebar
   - See chunk count for each document
   - Delete documents individually with the 🗑️ button
   - Toggle RAG on/off as needed

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

- **API Key**: Required to use the chat functionality
- **Base URL**: Optional, used for custom endpoints (like Azure OpenAI)
- **Model Selection**: Choose from common models or enter a custom model name

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

## Notes

- Your API key is required to use the chat functionality
- The application uses gpt-4o-2024-11-20 by default
- Your prompts are saved locally in a JSON file
