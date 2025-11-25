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
from typing import List


load_dotenv()

app = FastAPI()

# ---------------------------------------------
# Markdown Processing
# ---------------------------------------------

def process_file(file_path: str):
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

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
  raise ValueError("GROQ_API_KEY missing in .env")

llm = ChatGroq(
  groq_api_key=groq_api_key,
  model="llama-3.1-8b-instant",
  temperature=0.2,
  max_tokens=1024,
)

# ---------------------------------------------
# Embedding + Vector Store (your existing implementations)
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

# ensure index for efficient retrieval (created at startup)

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


class CreateSessionRequest(BaseModel):
  user_id: str
  title: str | None = None


class ChatMessageRequest(BaseModel):
  user_id: str
  message: str
  top_k: int = 3
  memory_limit: int = 10  # last N messages

# ---------------------------------------------
# Helper: load memory from MongoDB
# ---------------------------------------------

async def load_memory_from_mongo(session_id: str, limit: int = 10) -> str:
  messages = await chat_messages \
    .find({"session_id": session_id}) \
    .sort("timestamp", -1) \
    .limit(limit) \
    .to_list(length=None)

  messages = list(reversed(messages))

  memory_text = ""
  for m in messages:
    role = m.get("role", "").upper()
    content = m.get("content", "")
    memory_text += f"{role}: {content}\n"

  return memory_text.strip()

# ---------------------------------------------
# Advanced RAG helper functions
# ---------------------------------------------

def rewrite_query(query: str) -> str:
  prompt = f"""
Rewrite the following query into a clearer and more explicit search query.
Query: "{query}"
"""
  resp = llm.invoke(prompt)
  return resp.content.strip()


def expand_query(query: str) -> List[str]:
  prompt = f"Generate up to 3 concise search query variations for: {query}\nReturn each on a new line." 
  resp = llm.invoke(prompt).content.strip()
  lines = [l.strip() for l in resp.split("\n") if l.strip()]
  return lines[:3]


def rerank_docs(query: str, docs: List[dict]) -> List[dict]:
  # Simple LLM-based reranking. For performance, limit the number of docs.
  scored = []
  for d in docs[:20]:
    score_prompt = f"""
Rate the relevance from 0 to 10 (integer) of the following text for answering the query:
Query: {query}

Text:
{d.get('content','')}

Only return the integer score.
"""
    try:
      score_resp = llm.invoke(score_prompt).content.strip()
      score = int(''.join(ch for ch in score_resp if ch.isdigit()) or 0)
    except Exception:
      score = 0

    scored.append((score, d))

  scored.sort(reverse=True, key=lambda x: x[0])
  return [d for s, d in scored]


def summarize_docs(docs: List[dict]) -> str:
  if not docs:
    return ""

  combined = "\n\n".join(d.get("content", "") for d in docs[:6])
  prompt = f"Summarize the following content into a concise useful summary that helps answer a user's question:\n\n{combined}"
  resp = llm.invoke(prompt).content.strip()
  return resp


def fuse_context(summary: str, docs: List[dict]) -> str:
  out = ""
  if summary:
    out += f"SUMMARY:\n{summary}\n\n"
  out += "TOP_SNIPPETS:\n"
  for d in docs[:5]:
    src = d.get("metadata", {}).get("source", "")
    snippet = d.get("content", "")[:800]
    out += f"- ({src}) {snippet}\n\n"
  return out

# ---------------------------------------------
# Advanced retrieval orchestration
# ---------------------------------------------

def advanced_retrieve(query: str, retriever: RAGRetriever, top_k: int = 3) -> List[dict]:
  # 1. Pre-retrieval: rewrite + expand
  rewritten = rewrite_query(query)
  expansions = expand_query(rewritten)

  queries = [rewritten] + expansions

  # 2. Retrieval for each query
  all_docs = []
  for q in queries:
    try:
      docs = retriever.retrieve(q, top_k=top_k)
      all_docs.extend(docs)
    except Exception as e:
      print(f"Retrieval error for query '{q}': {e}")

  # deduplicate by content
  seen = set()
  unique_docs = []
  for d in all_docs:
    key = (d.get('content') or '')[:400]
    if key in seen:
      continue
    seen.add(key)
    unique_docs.append(d)

  # 3. Post-retrieval: rerank + summarize + fusion
  reranked = rerank_docs(query, unique_docs)
  summary = summarize_docs(reranked)
  fused = fuse_context(summary, reranked)

  # return fused context plus top docs for traceability
  return {"fused": fused, "top_docs": reranked[: top_k * 2]}

# ---------------------------------------------
# RAG Answering (advanced)
# ---------------------------------------------

def build_prompt(memory_text: str, fused_context: str, user_query: str) -> str:
  return f"""
You are an advanced RAG assistant. Use the conversation memory and the retrieved knowledge to answer concisely and cite sources when possible.

Conversation History:
{memory_text}

Retrieved Knowledge:
{fused_context}

User Question:
{user_query}

Assistant:
"""

# ---------------------------------------------
# Startup Event -> Load Docs & Vector Store
# ---------------------------------------------
@app.on_event("startup")
async def startup_event():
  global rag_loaded

  # create index for message collection (improves query performance)
  try:
    await chat_messages.create_index([("session_id", 1), ("timestamp", -1)])
  except Exception as e:
    print(f"Index creation skipped or failed: {e}")

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
      "googleId": data.google_id,
    }
    result = await users.insert_one(new_user)
    new_user["_id"] = str(result.inserted_id)
    user = new_user
  else:
    user["_id"] = str(user["_id"])  # Convert ObjectId

  access_token = create_access_token({"user_id": user["_id"], "email": user["email"]})

  return {"accessToken": access_token, "user": user}

@app.post("/chat/session")
async def create_chat_session(data: CreateSessionRequest):
  session = {
    "_id": str(uuid.uuid4()),
    "user_id": data.user_id,
    "title": data.title or "New Chat",
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow(),
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
  return {"deleted": True}

async def save_message(session_id: str, user_id: str, role: str, content: str):
  message = {
    "_id": str(uuid.uuid4()),
    "session_id": session_id,
    "user_id": user_id,
    "role": role,
    "content": content,
    "timestamp": datetime.utcnow(),
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

  # 2. Save user message immediately (so it becomes part of history)
  await save_message(session_id, data.user_id, "user", data.message)

  # 3. Load memory text (after saving so user message is included)
  memory_text = await load_memory_from_mongo(session_id, limit=data.memory_limit)

  # 4. Advanced retrieval (pre-retrieval + retrieval + post-retrieval)
  retriever = RAGRetriever(vector_store=vectorstore, embedding_manager=embeddings_manager)
  retrieval_out = advanced_retrieve(data.message, retriever, top_k=data.top_k)
  fused_context = retrieval_out.get("fused", "")
  top_docs = retrieval_out.get("top_docs", [])

  # 5. Build prompt
  prompt = build_prompt(memory_text, fused_context, data.message)

  # 6. Generate answer
  response = llm.invoke(prompt)
  answer = response.content.strip()

  # 7. Save assistant message
  await save_message(session_id, data.user_id, "assistant", answer)

  # 8. Return response with traceability info
  return {
    "session_id": session_id,
    "user_message": data.message,
    "assistant_answer": answer,
    "retrieved": {"count": len(top_docs), "top_docs": [{"source": d.get('metadata',{}).get('source'), "snippet": d.get('content','')[:400]} for d in top_docs]},
  }
