Document Chatbot 💬
This is a chatbot application built with Streamlit that allows you to chat with your own documents. You can upload PDFs, DOCX files, text files, or markdown files, and the app will use a sophisticated AI agent (powered by LangChain and Groq) to answer your questions based on their content.

Powered by Streamlit, LangChain, Groq (Llama 3), FAISS, and Hugging Face Embeddings.

🚀 Features
Chat with Multiple Documents: Supports PDF, DOCX, TXT, and MD files.

Fast Responses: Uses the high-speed Groq API with the Llama 3 70b model.

Intelligent Retrieval: Employs a LangChain ReAct agent to intelligently decide when to search the documents for answers.

High-Quality Embeddings: Uses the popular all-MiniLM-L6-v2 model from Hugging Face for local document embedding.

Smart Caching: Automatically caches the processed document vector store (FAISS index) in a ./faiss_cache folder. Re-uploading the same files will load the cache instantly.

Default Documents: Can be pre-loaded with documents from a ./default_documents folder on startup.

Web Interface: Clean, simple chat interface provided by Streamlit, including chat history and a "Clear Conversation" button.
