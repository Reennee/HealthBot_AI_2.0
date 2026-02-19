"""
HealthBot AI - FastAPI Inference Server
Serves the fine-tuned healthcare LLM for chat inference.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="HealthBot AI",
    description="Healthcare chatbot powered by a fine-tuned LLM",
    version="1.0.0",
)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": False}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # TODO: Load fine-tuned model and generate response (Step 7)
    return ChatResponse(
        response="[Placeholder] Model not loaded yet. Complete Step 7 to enable inference."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
