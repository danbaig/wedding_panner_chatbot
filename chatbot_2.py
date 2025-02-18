import os
import json
from dotenv import load_dotenv
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# LangChain & LangGraph imports
from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langsmith import traceable

# LangGraph and tool-related imports
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage as LHumanMessage
from graph_state import AgentState, AgentAction

# ---------------- Load Environment Variables ----------------
load_dotenv()

# ---------------- Constants ----------------
CHROMA_DB_PATH = "./chroma_db_json"
JSON_FILE = "./data/vendors_live_records_v2.json"
IMPORTANT_CATEGORIES = ["Wedding Venues", "Photographers", "Caterers", "Transportation"]

# ---------------- Initialize Embeddings, LLM & Memory ----------------

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0
)
memory = ConversationBufferMemory(return_messages=True)

# ---------------- Helper: Get or Build the Chroma Vector Store ----------------
def get_chroma_instance():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name="documents",
        embedding_function=EMBEDDING_MODEL
    )

def load_vendors(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("vendors", [])

def vendors_to_documents(vendors: list) -> list:
    docs = []
    for vendor in vendors:
        # Store the full vendor details as JSON.
        docs.append(Document(page_content=json.dumps(vendor), metadata={"id": vendor.get("id")}))
    return docs

vendors = load_vendors(JSON_FILE)
documents = vendors_to_documents(vendors)

# (Optional) Split documents into chunks for better retrieval.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = splitter.split_documents(documents)
texts = [doc.page_content for doc in doc_chunks]
metadatas = [doc.metadata for doc in doc_chunks]

vectorstore = get_chroma_instance()
if not vectorstore._collection.get()["ids"]:
    vectorstore.add_texts(texts, metadatas=metadatas)
    vectorstore.persist()
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# ---------------- SQL Data: Budget Categories and Items ----------------
budget_categories = [
    {"id": 2, "name": "Apparel"},
    {"id": 3, "name": "Reception"},
    {"id": 4, "name": "Music or Entertainment"},
    {"id": 5, "name": "Printing or Stationery"},
    {"id": 6, "name": "Photography"},
    {"id": 7, "name": "Decorations"},
    {"id": 8, "name": "Flowers"},
    {"id": 9, "name": "Gifts"},
    {"id": 10, "name": "Travel or Transportation"},
    {"id": 11, "name": "Other"}
]

budget_category_items = [
    {"id": 1, "budget_category_id": 2, "name": "Engagement ring(s)", "ratio_value": 0.030},
    {"id": 3, "budget_category_id": 2, "name": "Spouse-to-be 1 ring", "ratio_value": 0.061},
    {"id": 4, "budget_category_id": 2, "name": "Spouse-to-be 1 gown", "ratio_value": 0.076},
    {"id": 5, "budget_category_id": 2, "name": "Spouse-to-be 1 veil/headpiece", "ratio_value": 0.015},
    {"id": 6, "budget_category_id": 2, "name": "Spouse-to-be 1 shoes", "ratio_value": 0.005},
    {"id": 7, "budget_category_id": 2, "name": "Spouse-to-be 1 jewelry", "ratio_value": 0.012},
    {"id": 8, "budget_category_id": 2, "name": "Spouse-to-be 1 hosiery", "ratio_value": 0.001},
    {"id": 9, "budget_category_id": 2, "name": "Spouse-to-be 2 ring", "ratio_value": 0.009},
    {"id": 10, "budget_category_id": 2, "name": "Spouse-to-be 2 tuxedo", "ratio_value": 0.006},
    {"id": 11, "budget_category_id": 2, "name": "Spouse-to-be 2 veil/headpiece", "ratio_value": 0.000},
    {"id": 12, "budget_category_id": 2, "name": "Spouse-to-be 2 shoes", "ratio_value": 0.002},
    {"id": 13, "budget_category_id": 2, "name": "Spouse-to-be 2 jewelry", "ratio_value": 0.000},
    {"id": 14, "budget_category_id": 2, "name": "Spouse-to-be 2 hosiery", "ratio_value": 0.001},
    {"id": 15, "budget_category_id": 3, "name": "Room/hall fees", "ratio_value": 0.076},
    {"id": 16, "budget_category_id": 3, "name": "Tables and chairs", "ratio_value": 0.000},
    {"id": 17, "budget_category_id": 3, "name": "Food", "ratio_value": 0.303},
    {"id": 18, "budget_category_id": 3, "name": "Drinks", "ratio_value": 0.030},
    {"id": 19, "budget_category_id": 3, "name": "Linens", "ratio_value": 0.000},
    {"id": 20, "budget_category_id": 3, "name": "Cake", "ratio_value": 0.012},
    {"id": 21, "budget_category_id": 3, "name": "Favors", "ratio_value": 0.006},
    {"id": 22, "budget_category_id": 3, "name": "Staff and gratuities", "ratio_value": 0.006},
    {"id": 23, "budget_category_id": 4, "name": "Musicians for ceremony", "ratio_value": 0.000},
    {"id": 24, "budget_category_id": 4, "name": "Band/DJ for reception", "ratio_value": 0.045},
    {"id": 25, "budget_category_id": 5, "name": "Invitations", "ratio_value": 0.012},
    {"id": 26, "budget_category_id": 5, "name": "Announcements", "ratio_value": 0.003},
    {"id": 27, "budget_category_id": 5, "name": "Thank-You cards", "ratio_value": 0.003},
    {"id": 28, "budget_category_id": 5, "name": "Personal stationery", "ratio_value": 0.000},
    {"id": 29, "budget_category_id": 5, "name": "Guest book", "ratio_value": 0.001},
    {"id": 30, "budget_category_id": 5, "name": "Programs", "ratio_value": 0.002},
    {"id": 31, "budget_category_id": 5, "name": "Reception napkins", "ratio_value": 0.001},
    {"id": 32, "budget_category_id": 5, "name": "Matchbooks", "ratio_value": 0.000},
    {"id": 33, "budget_category_id": 5, "name": "Calligraphy", "ratio_value": 0.000},
    {"id": 34, "budget_category_id": 6, "name": "Photography", "ratio_value": 0.076},
    {"id": 35, "budget_category_id": 6, "name": "Extra prints", "ratio_value": 0.001},
    {"id": 36, "budget_category_id": 6, "name": "Photo albums", "ratio_value": 0.003},
    {"id": 37, "budget_category_id": 6, "name": "Videography", "ratio_value": 0.045},
    {"id": 38, "budget_category_id": 7, "name": "Bows for seating", "ratio_value": 0.000},
    {"id": 39, "budget_category_id": 7, "name": "Centerpieces", "ratio_value": 0.009},
    {"id": 40, "budget_category_id": 7, "name": "Candles", "ratio_value": 0.003},
    {"id": 41, "budget_category_id": 7, "name": "Lighting", "ratio_value": 0.003},
    {"id": 42, "budget_category_id": 7, "name": "Balloons", "ratio_value": 0.006},
    {"id": 43, "budget_category_id": 8, "name": "Bouquets", "ratio_value": 0.012},
    {"id": 44, "budget_category_id": 8, "name": "Boutonnières", "ratio_value": 0.005},
    {"id": 45, "budget_category_id": 8, "name": "Corsages", "ratio_value": 0.000},
    {"id": 46, "budget_category_id": 8, "name": "Ceremony", "ratio_value": 0.045},
    {"id": 47, "budget_category_id": 8, "name": "Reception", "ratio_value": 0.000},
    {"id": 48, "budget_category_id": 9, "name": "Attendants", "ratio_value": 0.014},
    {"id": 49, "budget_category_id": 9, "name": "Spouse-to-be 1", "ratio_value": 0.000},
    {"id": 50, "budget_category_id": 9, "name": "Spouse-to-be 2", "ratio_value": 0.000},
    {"id": 51, "budget_category_id": 9, "name": "Parents", "ratio_value": 0.000},
    {"id": 52, "budget_category_id": 9, "name": "Readers/other participants", "ratio_value": 0.000},
    {"id": 53, "budget_category_id": 10, "name": "Limousines/trolleys", "ratio_value": 0.012},
    {"id": 54, "budget_category_id": 10, "name": "Parking", "ratio_value": 0.000},
    {"id": 55, "budget_category_id": 10, "name": "Taxis", "ratio_value": 0.000},
    {"id": 56, "budget_category_id": 11, "name": "Officiant", "ratio_value": 0.024},
    {"id": 57, "budget_category_id": 11, "name": "Church/ceremony site fee", "ratio_value": 0.000},
    {"id": 58, "budget_category_id": 11, "name": "Wedding coordinator", "ratio_value": 0.000},
    {"id": 59, "budget_category_id": 11, "name": "Rehearsal dinner", "ratio_value": 0.014},
    {"id": 60, "budget_category_id": 11, "name": "Engagement party", "ratio_value": 0.000},
    {"id": 61, "budget_category_id": 11, "name": "Showers", "ratio_value": 0.000},
    {"id": 62, "budget_category_id": 11, "name": "Salon appointments", "ratio_value": 0.001},
    {"id": 63, "budget_category_id": 11, "name": "Bachelor/ette parties", "ratio_value": 0.000},
    {"id": 64, "budget_category_id": 11, "name": "Brunch", "ratio_value": 0.009},
    {"id": 65, "budget_category_id": 11, "name": "Hotel rooms", "ratio_value": 0.000}
]

def parse_budget(budget_str: str) -> float:
    s = budget_str.lower().strip().replace(",", "")
    if s.endswith("k"):
        try:
            return float(s[:-1]) * 1000
        except:
            return 0.0
    try:
        return float(s)
    except:
        return 0.0

# ---------------- Define Tools as Discrete Functions ----------------

@tool("conversational_chain")
def conversational_chain(query: str) -> str:
    """
    Uses the conversation history and vendor context to generate a general answer.
    """
    memory_vars = memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", "")
    refined_query = f"{chat_history}\nNew Query: {query}" if chat_history else query
    context_docs = retriever.invoke(refined_query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt_template = (
        "You are Sara, the wedding planning chatbot. Answer the user's query in a friendly, playful tone using the wedding details and vendor context below.\n\n"
        "Wedding Details:\n"
        "  - Name: {name}\n"
        "  - Partner: {spouse}\n"
        "  - Wedding Location: {location}\n"
        "  - Budget: {budget}\n"
        "  - Date: {date}\n\n"
        "User Query: {user_query}\n\n"
        "Vendor Context: {document_context}\n\n"
        "Answer:"
    )
    details = st.session_state.wedding_details if "wedding_details" in st.session_state else {
        "name": "Guest", "spouse": "Partner", "location": "Unknown", "budget": "Not specified", "date": "Not specified"
    }
    full_prompt = prompt_template.format(
        name=details["name"],
        spouse=details["spouse"],
        location=details["location"],
        budget=details["budget"],
        date=details.get("date", "Not specified"),
        user_query=query,
        document_context=context_text
    )
    response = llm.invoke([LHumanMessage(content=full_prompt)])
    return response.content

@tool("vendor_retrieval")
def vendor_retrieval(query: str) -> str:
    """
    Retrieves and formats vendor information based on the query.

    This function uses the vector store retriever to obtain documents (each containing a full vendor JSON)
    and then extracts only the title and profile_url fields. It returns a markdown table with two columns:
    Vendor Name and Profile URL.
    
    :param query: The search query provided by the user.
    :return: A formatted markdown table with the vendor details.
    """
    context_docs = retriever.invoke(query)
    if context_docs:
        rows = []
        for doc in context_docs:
            try:
                data = json.loads(doc.page_content)
                title = data.get("title", "Unknown")
                profile_url = data.get("profile_url", "No URL")
                # Create a table row for each vendor.
                rows.append(f"| {title} | {profile_url} |")
            except Exception as e:
                print("EXCEPTION"*10, e)
                continue
        if rows:
            # Create the markdown table header.
            header = "| Vendor Name | Profile URL |\n| --- | --- |\n"
            table = header + "\n".join(rows)
            followup = "Would you like more details or to explore other services?"
            return f"Here's what I found for your query:\n\n{table}\n\n{followup}"
        else:
            return "I'm sorry, I couldn't locate any vendor profiles at the moment."
    else:
        return "I couldn't find any matching vendors at the moment. Would you like to try a different service?"


@tool("ask_followup")
def ask_followup() -> str:
    """
    Asks the user to specify which wedding service they're interested in.
    """
    return "Sara: Could you please specify which wedding service you're interested in? For example, are you looking for venues, photographers, or caterers?"

@tool("irrelevant_query_detector")
def irrelevant_query_detector(query: str) -> str:
    """
    Informs the user that their query appears off-topic.
    """
    return "Sara: That doesn't seem wedding-related. I'm here to help you plan your big day—shall we talk about venues, photographers, or caterers?"

@tool("final_answer")
def final_answer(answer: str) -> str:
    """
    Returns the final answer.
    """
    return f"Sara: {answer}"

# Budgeting Agent Tool
@tool("budgeting_agent")
def budgeting_agent(query: str) -> str:
    """
    Creates a budget breakdown for the client.
    
    If the user query is general (no specific category mentioned), it allocates the total budget across all
    categories based on the sum of item ratios in each category.
    
    If the query mentions a specific category (matched by name), it then breaks down the allocated budget
    for that category into its items based on the item ratios.
    
    The total budget is read from st.session_state.wedding_details["budget"].
    """
    total_budget_str = st.session_state.wedding_details.get("budget", "0")
    total_budget = parse_budget(total_budget_str)
    query_lower = query.lower()
    
    # Check for a specific category match.
    specific_category = None
    for cat in budget_categories:
        if cat["name"].lower() in query_lower:
            specific_category = cat
            break
    
    if specific_category:
        # Breakdown for the specific category.
        items = [item for item in budget_category_items if item["budget_category_id"] == specific_category["id"]]
        ratio_sum = sum(item["ratio_value"] for item in items)
        if ratio_sum == 0:
            return f"Sara: I'm sorry, I couldn't compute the breakdown for {specific_category['name']}."
        # Overall allocation for this category is computed based on the proportion of its items' ratios to the total.
        total_ratio = sum(item["ratio_value"] for item in budget_category_items)
        category_budget = total_budget * (sum(item["ratio_value"] for item in items) / total_ratio)
        rows = []
        for item in items:
            item_budget = category_budget * (item["ratio_value"] / ratio_sum)
            rows.append(f"| {item['name']} | {item_budget:,.2f} |")
        if rows:
            header = f"| Item | Allocated Budget for {specific_category['name']} |\n| --- | --- |\n"
            table = header + "\n".join(rows)
            return (f"Sara: For the **{specific_category['name']}** category, based on your total budget of {total_budget:,.2f}, "
                    f"we have allocated approximately {category_budget:,.2f}. Here's the breakdown:\n\n{table}")
        else:
            return f"Sara: I couldn't find any items for the category **{specific_category['name']}**."
    else:
        # General budgeting: breakdown for all categories based on item ratios.
        category_ratios = {}
        for cat in budget_categories:
            items = [item for item in budget_category_items if item["budget_category_id"] == cat["id"]]
            category_ratios[cat["name"]] = sum(item["ratio_value"] for item in items)
        total_ratio = sum(category_ratios.values())
        if total_ratio == 0:
            return "Sara: I'm sorry, I couldn't compute a general budget breakdown at the moment."
        rows = []
        for cat_name, ratio in category_ratios.items():
            allocated = total_budget * (ratio / total_ratio)
            rows.append(f"| {cat_name} | {allocated:,.2f} |")
        if rows:
            header = "| Category | Allocated Budget |\n| --- | --- |\n"
            table = header + "\n".join(rows)
            return (f"Sara: Based on your total budget of {total_budget:,.2f}, here's how it would break down across categories:\n\n"
                    f"{table}\n\nIf you'd like a detailed breakdown for any category, just let me know!")
        else:
            return "Sara: I'm sorry, I couldn't generate a budget breakdown at the moment."

# ---------------- Oracle and State Graph Setup ----------------

def oracle_wedding(query: str) -> str:
    query_lower = query.lower()
    # Check if query contains any budget category name as substring.
    for cat in budget_categories:
        if cat["name"].lower() in query_lower:
            return "budgeting_agent"
    # Also, if the query contains the word "budget", route to budgeting agent.
    if "budget" in query_lower:
        return "budgeting_agent"
    
    details = st.session_state.get("wedding_details", {
        "name": "Guest", "spouse": "Partner", "location": "Unknown",
        "budget": "Not specified", "date": "Not specified"
    })
    prompt = (
        "You are Sara, the wedding planning chatbot. Based on the conversation and the user's query, decide the next action in a playful tone.\n"
        "Wedding Details:\n"
        f"  - Name: {details['name']}\n"
        f"  - Partner: {details['spouse']}\n"
        f"  - Wedding Location: {details['location']}\n"
        f"  - Budget: {details['budget']}\n"
        f"  - Date: {details.get('date', 'Not specified')}\n\n"
        "Then reply with exactly one of these actions: ask_followup, vendor_retrieval, conversational_chain, budgeting_agent, or irrelevant_query_detector.\n"
        f"User Query: {query}\n"
        "Answer:"
    )
    response = llm.invoke([LHumanMessage(content=prompt)])
    decision_text = response.content.strip().lower()
    if "ask_followup" in decision_text or "more details" in decision_text:
        return "ask_followup"
    elif "vendor_retrieval" in decision_text or "retrieve" in decision_text:
        return "vendor_retrieval"
    elif "budgeting_agent" in decision_text:
        return "budgeting_agent"
    elif "irrelevant" in decision_text:
        return "irrelevant_query_detector"
    else:
        return "conversational_chain"


@traceable
def run_oracle(state: dict):
    user_query = state.get("input", "")
    decision = oracle_wedding(user_query)
    action = AgentAction(
        tool=decision,
        tool_input=user_query,
        log="Oracle decision: " + decision
    )
    return {"intermediate_steps": [action]}

def run_tool(state: dict):
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    tool_map = {
        "conversational_chain": conversational_chain,
        "vendor_retrieval": vendor_retrieval,
        "ask_followup": ask_followup,
        "irrelevant_query_detector": irrelevant_query_detector,
        "final_answer": final_answer,
        "budgeting_agent": budgeting_agent  # Ensure budgeting agent is included.
    }
    output = tool_map[tool_name].invoke(input=tool_args if tool_args else "")
    action = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(output)
    )
    return {"intermediate_steps": [action]}

def router(state: dict):
    if state.get("intermediate_steps"):
        return state["intermediate_steps"][-1].tool
    return "final_answer"

graph = StateGraph(AgentState)
graph.add_node("oracle", run_oracle)
graph.add_node("vendor_retrieval", run_tool)
graph.add_node("ask_followup", run_tool)
graph.add_node("conversational_chain", run_tool)
graph.add_node("irrelevant_query_detector", run_tool)
graph.add_node("final_answer", run_tool)
graph.add_node("budgeting_agent", run_tool)  # Add budgeting_agent node.
graph.set_entry_point("oracle")
graph.add_conditional_edges(source="oracle", path=router)
graph.add_edge("vendor_retrieval", "final_answer")
graph.add_edge("ask_followup", "final_answer")
graph.add_edge("conversational_chain", "final_answer")
graph.add_edge("irrelevant_query_detector", "final_answer")
graph.add_edge("budgeting_agent", "final_answer")  # Route budgeting_agent to final answer.
graph.add_edge("final_answer", END)
runnable = graph.compile()

def update_chat_history(user_input, assistant_response):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

def wedding_chatbot_response(user_input: str) -> str:
    state = {
        "input": user_input,
        "chat_history": st.session_state.chat_history if "chat_history" in st.session_state else []
    }
    out = runnable.invoke(state)
    final_action = out.get("intermediate_steps", [])[-1]
    final_output = final_action.log
    return final_output

# ---------------- Session State Initialization ----------------

DETAILS_QUESTIONS = [
    {"key": "name", "question": "Sara: Hey there! I'm Sara, your wedding planner extraordinaire. What should I call you?"},
    {"key": "spouse", "question": "Sara: Awesome! And what's your partner's name?"},
    {"key": "location", "question": "Sara: Where's the big celebration taking place? (City or State)"},
    {"key": "budget", "question": "Sara: Great! What's your wedding budget?"},
    {"key": "date", "question": "Sara: And when's your big day?"}
]

if "phase" not in st.session_state:
    st.session_state.phase = "collect_details"
if "detail_index" not in st.session_state:
    st.session_state.detail_index = 0
if "wedding_details" not in st.session_state:
    st.session_state.wedding_details = {"name": "", "spouse": "", "location": "", "budget": "", "date": ""}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.phase == "collect_details" and not st.session_state.chat_history:
    current_question = DETAILS_QUESTIONS[st.session_state.detail_index]["question"]
    st.session_state.chat_history.append({"role": "assistant", "content": current_question})
elif st.session_state.phase == "chatting" and not st.session_state.chat_history:
    st.session_state.chat_history.append({"role": "assistant", "content": "Sara: Hi again! Ready to explore some wedding services?"})

st.title("Wedding Planner Chatbot with Multi-Agent Flow (GPT-4)")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**{msg['content']}**")

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def process_input():
    user_text = st.session_state.user_input.strip()
    if not user_text:
        return
    if st.session_state.phase == "collect_details":
        idx = st.session_state.detail_index
        key = DETAILS_QUESTIONS[st.session_state.detail_index]["key"]
        st.session_state.wedding_details[key] = user_text
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        st.session_state.detail_index += 1
        if st.session_state.detail_index < len(DETAILS_QUESTIONS):
            next_question = DETAILS_QUESTIONS[st.session_state.detail_index]["question"]
            st.session_state.chat_history.append({"role": "assistant", "content": next_question})
        else:
            st.session_state.phase = "chatting"
            welcome_msg = (
                "Sara: Fantastic, I've got all your details now! I can help you with an array of services like stunning venues, "
                "top-notch photographers, delicious catering, awesome DJs, and more. Which one would you like to explore first, "
                "or would you like a personalized recommendation?"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        response = wedding_chatbot_response(user_text)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        memory.save_context({"input": user_text}, {"output": response})
    st.session_state.user_input = ""

st.text_input("", key="user_input", placeholder="Type your response here...", on_change=process_input)

st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-bottom: 150px;
    }
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 10px;
        border-top: 1px solid #ddd;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="fixed-input"></div>', unsafe_allow_html=True)
