# AI Chat System with Custom System Prompts

A simple Streamlit application that allows you to:
- Create and manage multiple system prompts
- Chat with an AI using the selected system prompt
- Save and load your custom prompts
- Use custom API endpoints and models

## Features

- **Multiple System Prompts**: Create, edit, and delete custom system prompts
- **Chat Interface**: Interactive chat UI similar to ChatGPT
- **Model Selection**: Choose from various AI models or enter custom model names
- **API Flexibility**: Works with OpenAI, Azure OpenAI, and other compatible APIs
- **Persistence**: Save and load your prompts between sessions
- **API Key Management**: Securely enter your API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/automateAI.git
   cd automateAI
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   
   Copy the example template:
   ```
   cp env.example .env
   ```
   
   Edit the `.env` file with your credentials:
   ```
   # Required
   OPENAI_API_KEY=your_api_key_here
   
   # Optional
   OPENAI_API_BASE=https://your-endpoint.openai.azure.com
   OPENAI_MODEL=gpt-4o-2024-11-20
   ```

## Usage

1. Run the application:
   ```
   streamlit run demo.py
   ```

2. Configure your API:
   - Enter your API key
   - Optionally set a custom API base URL (for Azure or other endpoints)
   - Verify your API key to discover available models

3. Select or customize your model:
   - Use the default model (gpt-4o-2024-11-20)
   - Select from common models in the dropdown
   - Enter a custom model name or deployment ID

4. Use the pre-configured system prompts or create your own

5. Start chatting with the AI using your selected system prompt

## Features Guide

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
- Streamlit >= 1.45.0
- OpenAI Python library >= 1.6.0

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

### Features

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