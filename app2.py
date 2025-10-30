import gradio as gr
import sqlite3
import google.generativeai as genai
import os

# --- Configuration ---
# Set your API key as an environment variable for security
# In your terminal: export GEMINI_API_KEY="your_api_key_here"
# On Windows: set GEMINI_API_KEY="your_api_key_here"
API_KEY = os.environ.get("")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the app.")

genai.configure(api_key=API_KEY)

# Initialize the Gemini model
# Note: Changed model name to a standard one.
model = genai.GenerativeModel("gemini-1.5-flash")

DB_NAME = "chat_history.db"

# --- Database Functions ---

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_message(thread_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content)
    )
    conn.commit()
    conn.close()

def load_threads():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT thread_id FROM messages ORDER BY thread_id DESC") # Show newest first
    thread_ids = [row[0] for row in c.fetchall()]
    threads = []
    for tid in thread_ids:
        c.execute("SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id", (tid,))
        messages = [{"role": role, "content": content} for role, content in c.fetchall()]
        if messages: # Only add threads that have messages
            threads.append({"thread_id": tid, "messages": messages})
    conn.close()
    return threads

def get_next_thread_id():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT MAX(thread_id) FROM messages")
    result = c.fetchone()
    max_id = result[0] if result[0] else 0
    conn.close()
    return max_id + 1

# --- Chatbot Logic ---

def call_gemini_chatbot(history):
    """
    Generates a response from Gemini API using the conversation history.
    Expects history in OpenAI/Gradio format: [{"role": "user", "content": "..."}, ...]
    """
    # Convert Gradio/OpenAI format to Gemini format
    gemini_history = []
    for msg in history:
        # Change "assistant" to "model" for Gemini API
        role = "model" if msg["role"] == "assistant" else msg["role"]
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    try:
        response = model.generate_content(gemini_history)
        bot_response = response.text.strip()
        if not bot_response:
            bot_response = "Hmm, I couldn't come up with a response. Could you please rephrase your question?"
    except Exception as e:
        print(f"Error calling Gemini API: {e}") # Log the error to the console
        bot_response = f"Sorry, something went wrong. Please check the logs."

    return bot_response

def format_chat_history(threads):
    """
    Formats the chat history into collapsible sections for each thread.
    """
    if not threads:
        return "No conversation yet."

    formatted = ""
    for idx, thread in enumerate(threads, 1):
        # Use the first user message as a title, or just "Thread X"
        first_user_msg = next((m['content'] for m in thread['messages'] if m['role'] == 'user'), f"Thread {thread['thread_id']}")
        title = (first_user_msg[:50] + '...') if len(first_user_msg) > 50 else first_user_msg

        thread_content = ""
        for msg in thread["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            thread_content += f"**{role}:** {msg['content']}\n\n"
        
        # Use thread_id for a unique summary title
        formatted += f"<details><summary><strong>{title} (Thread {thread['thread_id']})</strong></summary>\n\n{thread_content}</details>\n\n"

    return formatted

def reset_chat():
    """
    Starts a new chat thread.
    """
    new_thread_id = get_next_thread_id()
    # [chat, history_state, history_box, thread_id_state]
    return [], [], "No conversation yet.", new_thread_id

# Initialize the database
init_db()

# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <style>
            #new-chat-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }
            #history-box details {
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 8px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
            }
            #history-box summary {
                cursor: pointer;
                font-weight: bold;
            }
        </style>
        """
    )

    gr.Markdown("# Gemini 1.5 Flash Chatbot")

    new_chat_button = gr.Button("🆕 New Chat", elem_id="new-chat-btn", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            # FIX 1: Specify type="messages"
            chat = gr.Chatbot(
                label="Gemini Chat",
                type="messages", 
                bubble_full_width=False,
                avatar_images=(None, "https://placehold.co/100x100/blue/white?text=G") # User, Bot
            )
            user_input = gr.Textbox(label="Your Message", placeholder="Type here and press Send", scale=4)
            submit_button = gr.Button("Send", variant="primary", scale=1)
        with gr.Column(scale=1):
            gr.Markdown("### Chat History")
            history_box = gr.Markdown("No conversation yet.", elem_id="history-box")

    history_state = gr.State([])  # Current thread's messages
    thread_id_state = gr.State(get_next_thread_id())  # Current thread ID

    def respond(user_input, history, thread_id):
        # 1. Add user message to state
        history.append({"role": "user", "content": user_input})
        
        # 2. Get bot response (pass the full history)
        # We show a streaming placeholder
        chat.append({"role": "assistant", "content": "..."})
        yield {chat: history}

        bot_response = call_gemini_chatbot(history)
        
        # 3. Replace placeholder with actual response
        history.pop() # Remove the "..."
        history.append({"role": "assistant", "content": bot_response})
        
        # 4. Save both new messages to DB
        save_message(thread_id, "user", user_input)
        save_message(thread_id, "assistant", bot_response)

        # 5. Load all threads for the sidebar
        threads = load_threads()
        sidebar_md = format_chat_history(threads)
        
        # 6. Return values
        # FIX 2: Return `history` (list of dicts) directly to the chatbot
        # and also update the history_state
        yield {
            chat: history,
            history_state: history,
            history_box: sidebar_md,
            user_input: ""
        }

    submit_button.click(
        respond,
        inputs=[user_input, history_state, thread_id_state],
        outputs=[chat, history_state, history_box, user_input]
    )
    
    user_input.submit(
        respond,
        inputs=[user_input, history_state, thread_id_state],
        outputs=[chat, history_state, history_box, user_input]
    )

    new_chat_button.click(
        fn=reset_chat,
        # FIX 3: Make sure the button also clears the visual `chat` component
        outputs=[chat, history_state, history_box, thread_id_state]
    )

    # Load history on page load
    def load_sidebar_history():
        threads = load_threads()
        sidebar_md = format_chat_history(threads)
        return sidebar_md

    demo.load(
        fn=load_sidebar_history,
        outputs=[history_box]
    )

# FIX 4: Corrected the typo __name__ and __main__
if __name__ == "__main__":
    demo.launch(share=True)
