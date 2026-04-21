from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from google import genai
import re

# -----------------------------
# Configure Gemini API
# -----------------------------
client = genai.Client(api_key="AIzaSyCTB3Vm6kMNsm7oVn4O5wtA7nn_9cFFds8")

# -----------------------------
# Load embedding model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# -----------------------------
# Load reranker model
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# Load vector database
# -----------------------------
db = FAISS.load_local(
    "backend/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Conversation memory
# -----------------------------
chat_history = []

# -----------------------------
# Tax domain guard
# -----------------------------
tax_keywords = [
    "tax","income tax","itr","return","filing",
    "verification","e-verification","everification",
    "refund","tds","pan","aadhaar","assessment",
    "section","income tax act","income tax bill",
    "deduction","80c","80d","hra",
    "new regime","old regime","tax regime",
    "budget","budget 2025","budget 2026",
    "circular","cbdt",
    "scheme","filing of return",
    "salary","income","capital gains"
]


def is_tax_question(question):
    q = question.lower()
    return any(word in q for word in tax_keywords)


# -----------------------------
# Extract tax section numbers
# -----------------------------
def extract_section(query):
    match = re.search(r"\b\d+[a-zA-Z]?\b", query)
    return match.group(0) if match else None


# -----------------------------
# Rerank retrieved documents
# -----------------------------
def rerank_docs(question, docs, top_k=8):

    if not docs:
        return []

    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]


# -----------------------------
# Main chatbot function
# -----------------------------
def ask_tax_bot(question):

    if not is_tax_question(question):
        return "I can only answer questions related to Indian taxation."

    section = extract_section(question)

    # Retrieve documents
    docs = db.similarity_search(question, k=15)

    # Section filtering
    if section:
        filtered_docs = [
            d for d in docs if section.lower() in d.page_content.lower()
        ]
        if filtered_docs:
            docs = filtered_docs

    # Rerank documents
    docs = rerank_docs(question, docs, top_k=8)

    # Build context
    context = "\n\n".join([d.page_content for d in docs])

    if not context.strip():
        return "I cannot find this information in the tax documents."

    # Conversation history
    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # Prompt
    prompt = f"""
You are an expert assistant on Indian Income Tax.

Use ONLY the information from the provided context.

Rules:
1. If the question is YES/NO, answer Yes or No first.
2. Do NOT invent information.
3. If the answer is not in the context, say:
"I cannot find this information in the tax documents."

Conversation History:
{history_text}

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

        answer = response.text

        if not answer:
            answer = "I couldn't generate a response."

    except Exception as e:
        print(f"Gemini API error: {e}")
        answer = "The tax assistant service is currently unavailable."

    # Clean formatting
    answer = answer.replace("\n", " ")
    answer = " ".join(answer.split())

    chat_history.append((question, answer))

    return answer