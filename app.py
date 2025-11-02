import os
import streamlit as st
import time
import tempfile
import hashlib
import nest_asyncio

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub # To pull a pre-made agent prompt

# Apply the nest_asyncio patch (useful for environments like Streamlit)
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- UI Configuration ---
st.set_page_config(
    page_title="Chat-Bot",
    layout="wide"
)

st.title(" Chatbot ")
st.markdown("Powered by Groq Llama3 & HuggingFace Embeddings")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Document Loader Function
def load_file(file_path):
    """Load documents with appropriate loader depending on file type."""
    if file_path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path, mode='elements')
    return loader.load()

# Core RAG Function
def process_and_embed_documents(uploaded_files):
    """Loads, splits, embeds, and stores documents in a vector store."""
    with st.spinner("Processing your documents... Please wait."):
        # Create a unique hash for the uploaded files
        file_contents = "".join(sorted(f.getvalue().decode('latin-1') for f in uploaded_files))
        files_hash = hashlib.md5(file_contents.encode()).hexdigest()

        # Cached FAISS index path
        faiss_index_path = f"./faiss_cache/{files_hash}"

        if os.path.exists(faiss_index_path):
            st.info("Loading cached documents...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectors = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            st.success("✅ Cache loaded successfully!")
            return vectors

        with tempfile.TemporaryDirectory() as temp_dir:
            docs = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                docs.extend(load_file(file_path))

            if not docs:
                st.error("Could not load any documents. Please check your files.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectors = FAISS.from_documents(final_documents, embeddings)

            os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
            vectors.save_local(faiss_index_path)

    st.success("✅ Documents processed and cached successfully!")
    return vectors

def load_default_documents():
    """Load predefined default PDFs from default_documents folder."""
    default_path = "./default_documents"
    faiss_index_path = "./faiss_cache/default_index"

    if os.path.exists(faiss_index_path):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    docs = []
    for filename in os.listdir(default_path):
        if filename.endswith((".pdf", ".docx", ".txt", ".md")):
            docs.extend(load_file(os.path.join(default_path, filename)))

    if not docs:
        st.error("⚠️ No documents found in default_documents folder.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(final_documents, embeddings)

    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    vectors.save_local(faiss_index_path)

    return vectors
def create_agent_executor(vectors):
    """Creates a ReAct agent using the official prompt from LangChain Hub."""

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    retriever = vectors.as_retriever()

    def format_retriever_output(docs):
        """Formats the retriever's output into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    retriever_tool = Tool(
        name="document_retriever",
        func=lambda query: format_retriever_output(retriever.invoke(query)),
        description="Searches and returns information from the user's uploaded documents. ALWAYS use this tool for any questions about the provided context."
    )

    tools = [retriever_tool]

    # Pull the official ReAct chat prompt from the hub.
    # This prompt is guaranteed to have the correct structure and placeholders.
    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor
# --- Sidebar ---
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md']
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# --- Document Loading Logic ---
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if uploaded_files:
    st.session_state.vector = process_and_embed_documents(uploaded_files)
    st.session_state.agent_executor = create_agent_executor(st.session_state.vector)
    st.session_state.messages = []
    st.session_state.chat_history = []

elif st.session_state.vector is None: # Only load defaults if no files are uploaded
    st.session_state.vector = load_default_documents()
    if st.session_state.vector:
        st.session_state.agent_executor = create_agent_executor(st.session_state.vector)
        st.success("✅ Loaded default documents!")

# --- Main Chat Interface ---
if not st.session_state.vector:
    st.info("Upload your documents in the sidebar or add them to the default_documents folder.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.agent_executor is None:
        st.warning("Please upload documents or add them to the default_documents folder.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_executor = st.session_state.agent_executor

                stream = agent_executor.stream({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })

                response_text = st.write_stream(
                    (chunk.get('output', '') for chunk in stream if 'output' in chunk)
                )

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=str(response_text))
        ])