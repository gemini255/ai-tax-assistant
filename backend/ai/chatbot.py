import os
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from google import genai

# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

# -----------------------------
# Models
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# Vector DB
# -----------------------------
db = FAISS.load_local(
    "backend/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Memory
# -----------------------------
chat_history = []

# -----------------------------
# Domain Guard
# -----------------------------
tax_keywords = [
    "tax", "income tax", "itr", "return", "filing",
    "refund", "tds", "pan", "aadhaar",
    "deduction", "80c", "80d", "hra",
    "new regime", "old regime", "tax regime",
    "section", "salary", "income",
    "capital gains", "gst", "rebate"
]

def is_tax_question(question: str):
    q = question.lower()
    return any(word in q for word in tax_keywords)

# -----------------------------
# Extract section number
# -----------------------------
def extract_section(query):
    # catches 80c, 80 c, section 80c, 194j etc.
    match = re.search(r'(\d+\s*[a-zA-Z]?)', query.lower())
    if match:
        return match.group(1).replace(" ", "")
    return None

# -----------------------------
# Rerank
# -----------------------------
def rerank_docs(question, docs, top_k=4):
    if not docs:
        return []

    pairs = [(question, d.page_content[:1200]) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# -----------------------------
# Clean output
# -----------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Main chatbot
# -----------------------------
def ask_tax_bot(question):

    if not is_tax_question(question):
        return "I can only answer questions related to Indian taxation."

    section = extract_section(question)

    # Better retrieval
    docs = db.similarity_search(question, k=25)

    # Flexible section filter
    if section:
        filtered = []
        for d in docs:
            txt = d.page_content.lower()
            txt = txt.replace(" ", "").replace("\n", "")
            if section in txt:
                filtered.append(d)

        if filtered:
            docs = filtered

    # Rerank top docs
    docs = rerank_docs(question, docs, top_k=4)

    # Build context
    context = "\n\n".join([d.page_content[:1500] for d in docs])

    # If no context, fallback from top docs directly
    if not context.strip():
        if docs:
            return clean_text(docs[0].page_content[:600])
        return "I could not retrieve relevant tax information right now."

    # History
    history = ""
    for q, a in chat_history[-3:]:
        history += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are an expert assistant on Indian Income Tax.

Use ONLY the provided context.
Be clear, concise, and helpful.

Rules:
1. If yes/no question, answer Yes or No first.
2. If section asked, explain it clearly.
3. Use bullet points when useful.
4. Do not invent laws not in context.
5. If exact answer missing, give best available answer from context.

Conversation:
{history}

Context:
{context}

User Question:
{question}

Answer:
"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        answer = response.text if response.text else ""

        if not answer.strip():
            answer = docs[0].page_content[:600]

    except Exception:
        # Fallback if API fails
        answer = docs[0].page_content[:600]

    answer = clean_text(answer)

    chat_history.append((question, answer))

    return answer