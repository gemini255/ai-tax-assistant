from fastapi import FastAPI
from pydantic import BaseModel
from backend.tax_engine.suggest_regime import suggest_regime
from backend.ai.chatbot import ask_tax_bot

app = FastAPI()


# -----------------------------
# Request Models
# -----------------------------

class TaxRequest(BaseModel):
    income: float
    sec80c: float
    sec80d: float
    hra: float


class ChatRequest(BaseModel):
    question: str


# -----------------------------
# Home Route
# -----------------------------

@app.get("/")
def home():
    return {"message": "AI Tax Assistant Running"}


# -----------------------------
# Tax Calculation API
# -----------------------------

@app.post("/calculate-tax")
def calculate_tax(request: TaxRequest):

    income = request.income
    d80c = request.sec80c
    d80d = request.sec80d
    hra = request.hra

    result = suggest_regime(income, d80c, d80d, hra)

    return result


# -----------------------------
# Chatbot API
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question

    answer = ask_tax_bot(question)

    return {"response": answer}