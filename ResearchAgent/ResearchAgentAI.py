import json
import openai 
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun



# Title and Sidebar (API Key)
st.set_page_config(page_title="Research AI Agent")
st.title("Research AI Agent")

st.sidebar.title("🔍 Research Assistant")
st.markdown(
    """
    Welcome!
                
    This assistant can help you research topics using Wikipedia, and DuckDuckGo.
    """
)
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")


# Function Callings
functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Use this tool to search for a topic summary on Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term for Wikipedia"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Use this tool to search the web for information and return a quick summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Find information on the web"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "webscraper",
            "description": "Use this tool to scrape text content from a given webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_research_to_file",
            "description": "Use this tool to save result to a .txt file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Text content to save"}
                },
                "required": ["content"]
            }
        }
    }
]


# Tool Handlers 
def wikipedia_search(args):
    return WikipediaAPIWrapper().run(args["query"])

#### wrapped due to HTTP failures
def duckduckgo_search(args):
    try:
        return DuckDuckGoSearchAPIWrapper().run(args["query"])
    except Exception as e:
        return f"DuckDuckGo search failed: {str(e)}"


def webscraper(args):
    url = args["url"]
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    return "\n".join(p.get_text() for p in soup.find_all("p")[:10])

def save_research_to_file(args):
    with open("research.txt", "w", encoding="utf-8") as f:
        f.write(args["content"])
    return "Research saved to research.txt"

functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "webscraper": webscraper,
    "save_research_to_file": save_research_to_file,
}

# Other Functions 
def get_run(run_id, thread_id):
    return openai.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)

def send_message(thread_id, content):
    return openai.beta.threads.messages.create(thread_id=thread_id, role="user", content=content)

def get_messages(thread_id):
    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        st.chat_message(message.role).markdown(message.content[0].text.value)

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        outputs.append({
            "output": functions_map[function.name](json.loads(function.arguments)),
            "tool_call_id": action_id,
        })
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return openai.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )




# Streamlit 
if openai_api_key:
    openai.api_key = openai_api_key

    # Initialize Assistant 
    if "assistant" not in st.session_state:
        st.session_state.assistant = openai.beta.assistants.create(
            name="Research Assistant",
            instructions="You are a research expert. Use tools to gather accurate, sourced answers and save them to a file.",
            model="gpt-4o",
            tools=functions,
        )

    assistant_id = st.session_state.assistant.id

    # User Input
    user_input = st.text_input("Enter the research topic:")

    if st.button("Run") and user_input:
        thread = openai.beta.threads.create(
            messages=[{"role": "user", "content": user_input}]
        )

        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        with st.spinner("Processing ..."):
            while True:
                run_status = get_run(run.id, thread.id)

                if run_status.status == "completed":
                    break
                elif run_status.status == "requires_action":
                    submit_tool_outputs(run.id, thread.id)
      

        get_messages(thread.id)

        # Show download button for saved results
        try:
            with open("research.txt", "r", encoding="utf-8") as f:
                content = f.read()
            st.download_button(
                label="Download Research File",
                data=content,
                file_name="research.txt",
                mime="text/plain"
            )
        except FileNotFoundError:
            st.warning("The research file was not found. It may not have been saved properly.")
else:
    st.warning("Please enter your OpenAI API key to continue.")
