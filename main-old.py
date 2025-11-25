from fastapi import FastAPI, HTTPException
from db.mongodb import get_database
from pydantic import BaseModel
from helper.token import create_access_token
from fastapi.middleware.cors import CORSMiddleware
from embedding.embedding import EmbeddingManager
from vectorstore.vectorstore import VectorStore
from rag.rag_retrivel import RAGRetriever
from langchain_groq import ChatGroq
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv
import uuid
from datetime import datetime
from fastapi import Body


load_dotenv()

app = FastAPI()

# ---------------------------------------------
# Markdown Processing
# ---------------------------------------------
def process_file(file_path):
    all_documents = []
    md_dir = Path(file_path)
    md_files = list(md_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")

    for md_file in md_files:
        print(f"Processing file: {md_file}")

        try:
            loader = UnstructuredMarkdownLoader(str(md_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = md_file.name
                doc.metadata["file_type"] = md_file.suffix[1:]

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} docs from {md_file}")

        except Exception as e:
            print(f"Error loading file {md_file}: {e}")

    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} docs into {len(split_docs)} chunks")
    return split_docs

# ---------------------------------------------
# LLM Setup
# ---------------------------------------------
groq_api_key=os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY missing in .env")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024
)

# ---------------------------------------------
# RAG Answering Logic
# ---------------------------------------------
def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return "No relevant context found."

    context = "\n\n".join(doc["content"] for doc in results)

    prompt = f"""
You are an AI assistant. Use the context to answer concisely.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content.strip()

# ---------------------------------------------
# Embedding + Vector Store
# ---------------------------------------------
embeddings_manager = EmbeddingManager()
vectorstore = VectorStore()
rag_loaded = False

# ---------------------------------------------
# CORS
# ---------------------------------------------
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------
# Database
# ---------------------------------------------
db = get_database("domo-rag")
users = db["users"]
chat_sessions = db["chat_sessions"]
chat_messages = db["chat_messages"]

# ---------------------------------------------
# Schemas
# ---------------------------------------------
class GoogleAuthRequest(BaseModel):
    email: str
    name: str
    avatar: str
    google_id: str


class QueryRequest(BaseModel):
    query: str

# ---------------------------------------------
# Startup Event -> Load Docs & Vector Store
# ---------------------------------------------
@app.on_event("startup")
def startup_event():
    global rag_loaded

    if rag_loaded:
        return

    print("\n🚀 Loading documents and embeddings...")
    all_docs = process_file("./data")
    chunks = split_documents(all_docs)

    texts = [doc.page_content for doc in chunks]
    embeddings = embeddings_manager.generate_embedding(texts)

    vectorstore.add_documents(chunks, embeddings)

    rag_loaded = True
    print("✅ RAG system loaded successfully!\n")

# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.get("/")
def read_root():
    return {"status": "Running RAG Server"}

@app.post("/auth/google")
async def google_auth(data: GoogleAuthRequest):
    user = await users.find_one({"email": data.email})

    if not user:
        new_user = {
            "email": data.email,
            "name": data.name,
            "avatar": data.avatar,
            "googleId": data.google_id
        }
        result = await users.insert_one(new_user)
        new_user["_id"] = str(result.inserted_id)
        user = new_user
    else:
        user["_id"] = str(user["_id"])  # Convert ObjectId

    access_token = create_access_token({
        "user_id": user["_id"],
        "email": user["email"]
    })

    return {
        "accessToken": access_token,
        "user": user
    }


class CreateSessionRequest(BaseModel):
    user_id: str
    title: str | None = None

@app.post("/chat/session")
async def create_chat_session(data: CreateSessionRequest):
    session = {
        "_id": str(uuid.uuid4()),
        "user_id": data.user_id,
        "title": data.title or "New Chat",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    await chat_sessions.insert_one(session)

    return {"session": session}

@app.get("/chat/{user_id}/session")
async def get_chat_session(user_id: str):
    sessions = await chat_sessions.find({"user_id": user_id}).to_list(length=None)
    return {"sessions": sessions}

@app.get("/chat/{session_id}/messages")
async def get_messages(session_id: str):
    msgs = await chat_messages.find({"session_id": session_id}).to_list(None)
    return {"messages": msgs}

# delete session and chat
@app.delete("/chat/{session_id}")
async def delete_session(session_id: str):
    await chat_sessions.delete_one({"_id": session_id})
    await chat_messages.delete_many({"session_id": session_id})

class ChatMessageRequest(BaseModel):
    user_id: str
    message: str
    top_k: int = 3
    memory_limit: int = 10  # last N messages

async def save_message(session_id, user_id, role, content):
    message = {
        "_id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    }
    await chat_messages.insert_one(message)

@app.post("/chat/{session_id}/send")
async def send_message(session_id: str, data: ChatMessageRequest):
    # 1. Get last N messages for conversational memory
    prev_msgs = await chat_messages \
        .find({"session_id": session_id}) \
        .sort("timestamp", -1) \
        .limit(data.memory_limit) \
        .to_list(None)

    prev_msgs = list(reversed(prev_msgs))  # chronological order

    # 2. Save user message
    await save_message(session_id, data.user_id, "user", data.message)

    # 3. Fetch relevant docs from vectorstore (RAG)
    retriever = RAGRetriever(vector_store=vectorstore, embedding_manager=embeddings_manager)
    rag_results = retriever.retrieve(data.message, top_k=data.top_k)
    rag_context = "\n\n".join([doc["content"] for doc in rag_results])

    # 4. Build memory text
    memory_text = ""
    for m in prev_msgs:
        memory_text += f"{m['role'].upper()}: {m['content']}\n"

    # 5. Prompt for LLM
    prompt = f"""
        You are a helpful assistant. Use the conversation memory and context.

        Conversation History:
        {memory_text}

        Relevant Knowledge:
        {rag_context}

        User: {data.message}

        Assistant:
    """

    # 6. Generate answer
    response = llm.invoke(prompt)
    answer = response.content.strip()

    # 7. Save assistant message
    await save_message(session_id, data.user_id, "assistant", answer)

    # 8. Return response
    return {
        "session_id": session_id,
        "user_message": data.message,
        "assistant_answer": answer
    }
