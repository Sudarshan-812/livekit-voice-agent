import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api as livekit_api
from rag import extract_text_from_pdf, chunk_text, store_chunks

load_dotenv()

app = FastAPI(title="Voice AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    text = extract_text_from_pdf(contents)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

    chunks = chunk_text(text)
    store_chunks(chunks, source_name=file.filename)

    return {"filename": file.filename, "chunks_stored": len(chunks)}


@app.get("/get-token")
async def get_token(system_prompt: str = ""):
    room_name = f"room-{uuid.uuid4().hex[:8]}"
    token = (
        livekit_api.AccessToken(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_grants(livekit_api.VideoGrants(room_join=True, room=room_name))
        .with_identity("user")
        .with_metadata(system_prompt)
        .to_jwt()
    )
    return {
        "token": token,
        "url": os.getenv("LIVEKIT_URL"),
        "room": room_name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
