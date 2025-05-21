import streamlit as st
import openai
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configuration
st.set_page_config(page_title="AI Chat System", layout="wide")

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

# Save and load functions
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
    
    # API Settings
    st.markdown("### API Configuration")
    
    # API Base URL (for custom endpoints)
    api_base = st.text_input("API Base URL (Optional)", 
                           placeholder="https://my-endpoint.openai.azure.com", 
                           value=st.session_state.api_base if st.session_state.api_base else "",
                           help="Leave empty for default OpenAI. For Azure, use your deployment URL.")
    
    if (api_base != st.session_state.api_base) and api_base.strip():
        st.session_state.api_base = api_base.strip()
    elif not api_base.strip() and st.session_state.api_base:
        st.session_state.api_base = None
    
    # API Key input
    st.markdown("### API Key (Required)")
    st.markdown("For OpenAI, get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
    api_key = st.text_input("Enter your API Key", type="password", value=st.session_state.api_key)
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        
    # API key status indicator and verification
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.api_key:
            st.success("API Key provided")
        else:
            st.error("API Key required")
    
    with col2:
        if st.button("Verify Key"):
            if not st.session_state.api_key:
                st.error("No API key to verify")
            else:
                try:
                    # Create client with optional base URL
                    if st.session_state.api_base:
                        client = openai.OpenAI(
                            api_key=st.session_state.api_key,
                            base_url=st.session_state.api_base
                        )
                    else:
                        client = openai.OpenAI(api_key=st.session_state.api_key)
                    
                    # First try with the current selected model
                    try:
                        response = client.chat.completions.create(
                            model=st.session_state.model,
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=5
                        )
                        st.success(f"✅ API key is valid! Model '{st.session_state.model}' is accessible.")
                    except Exception as model_error:
                        # If the selected model fails, try to at least verify the API key is valid
                        if "model_not_found" in str(model_error) or "does not have access to model" in str(model_error):
                            # Try to determine which models are available
                            st.warning(f"⚠️ Your API key cannot access the model: '{st.session_state.model}'")
                            st.info("Let's try to find a model you can access...")
                            
                            # Try to get available models directly
                            try:
                                available_models = client.models.list()
                                model_ids = [model.id for model in available_models.data]
                                if model_ids:
                                    st.success(f"✅ Found {len(model_ids)} available models for your API key!")
                                    st.info("Available models: " + ", ".join(model_ids[:5]) + 
                                           ("..." if len(model_ids) > 5 else ""))
                                    
                                    # If we found models, set to the first available one
                                    if model_ids:
                                        st.session_state.model = model_ids[0]
                                        st.info(f"Set your model to: {model_ids[0]}")
                                else:
                                    st.warning("No models available with this API key.")
                            except Exception as list_error:
                                # If listing models failed, try some common alternatives
                                st.warning("Couldn't list models, trying common alternatives...")
                                
                                # Try common model variations
                                test_models = ["gpt-4", "gpt-4-turbo", "gpt-35-turbo", "text-davinci-003", 
                                             "claude-instant-1", "claude-2", "j2-light"]
                                for test_model in test_models:
                                    try:
                                        response = client.chat.completions.create(
                                            model=test_model,
                                            messages=[{"role": "user", "content": "Hello"}],
                                            max_tokens=5
                                        )
                                        st.success(f"✅ Found working model: '{test_model}'! Consider using this instead.")
                                        st.session_state.model = test_model
                                        break
                                    except:
                                        continue
                                else:
                                    # If no models worked, we still know the API key itself is valid
                                    st.error("❌ API key is valid but no standard models are accessible.")
                                    st.info("You may need to enter your specific deployment name or model ID.")
                        else:
                            # Some other error with the model
                            raise model_error
                            
                except openai.AuthenticationError:
                    st.error("❌ Invalid API key")
                except openai.RateLimitError:
                    st.warning("⚠️ Rate limit reached, but API key appears valid")
                except Exception as e:
                    if "invalid_api_key" in str(e):
                        st.error("❌ Invalid API key")
                    else:
                        st.error(f"❌ Error: {str(e)}")
        
    # Model selection
    st.subheader("Model Selection")
    custom_model = st.text_input("Enter model name (if yours isn't listed below)", 
                               placeholder="e.g., gpt-35-turbo or your custom deployment name")
    
    if custom_model and custom_model != st.session_state.model:
        st.session_state.model = custom_model
    
    model_options = [
        "gpt-4o-2024-11-20",  # Make this the first option
        "gpt-3.5-turbo", 
        "gpt-4", 
        "gpt-4-turbo",
        # Azure OpenAI common names
        "gpt-35-turbo",
        "gpt-4-32k",
        # Add some potential custom deployment names
        "text-davinci-003"
    ]
    
    if custom_model and custom_model not in model_options:
        model_options.append(custom_model)
        
    model = st.selectbox(
        "Or select a common model",
        options=model_options,
        index=0 if st.session_state.model not in model_options else model_options.index(st.session_state.model),
        help="Select the model to use. Your API key must have access to the selected model."
    )
    if model != st.session_state.model and not custom_model:
        st.session_state.model = model
        
    st.info(f"Currently using: **{st.session_state.model}**")
    
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
    
    # No need for the explicit check here anymore, it's handled by the callback
    
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
st.title("AI Chat System")
st.subheader(f"Active System Prompt: {st.session_state.selected_prompt}")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Function to generate AI responses
def generate_response(messages):
    if not st.session_state.api_key:
        return "⚠️ Please enter your OpenAI API key in the sidebar. You can get an API key from https://platform.openai.com/api-keys"
    
    try:
        # Create client with optional base URL
        if st.session_state.api_base:
            client = openai.OpenAI(
                api_key=st.session_state.api_key,
                base_url=st.session_state.api_base
            )
        else:
            client = openai.OpenAI(api_key=st.session_state.api_key)
        
        # Add system prompt
        full_messages = [
            {"role": "system", "content": st.session_state.system_prompts[st.session_state.selected_prompt]}
        ] + messages
        
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=full_messages,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    except openai.AuthenticationError:
        return "❌ **Authentication Error**: The API key you provided is invalid or has expired. Please check your API key and try again."
    except openai.RateLimitError:
        return "⚠️ **Rate Limit Error**: You've exceeded your current quota with OpenAI. Check your plan and billing details at https://platform.openai.com/account/billing"
    except openai.APIError as e:
        if "model_not_found" in str(e) or "does not have access to model" in str(e):
            return f"❌ **Model Access Error**: Your account doesn't have access to {st.session_state.model}. Try using gpt-3.5-turbo instead."
        return f"❌ **API Error**: {str(e)}"
    except Exception as e:
        return f"❌ **Error**: {str(e)}"

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.messages)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display model and timestamp info
st.caption(f"Using system prompt: {st.session_state.selected_prompt}")
st.caption(f"Model: {st.session_state.model}")
st.caption(f"Session time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
