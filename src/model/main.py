import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from rag_json import retrieve_documents, retrieve_documents_with_link
from dotenv import load_dotenv

load_dotenv()

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System Prompt desde system_prompt.md
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=4096,
    streaming=True,
)

class ChatRequest(BaseModel):
    prompt: str

def build_context_prompt(user_prompt: str, context: str) -> str:
    if not context:
        context_text = "No se encontró información relevante en la base de datos."
    else:
        context_text = f"Usa la siguiente información para responder:\n\n{context}"
    return f"{SYSTEM_PROMPT}\n\n{context_text}\n\nPregunta: {user_prompt}\nRespuesta:"

import re
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer, util

### MANEJO DE SALUDOS Y AGRADICIMIENTOS ###

# Inicializa el modelo
st_model = SentenceTransformer('all-MiniLM-L6-v2')

SALUDOS = ["hola", "buenos días", "buenas tardes", "buenas noches", "buen dia", "saludos"]
AGRADECIMIENTOS = ["gracias", "muchas gracias", "te agradezco", "agradecido", "agradezco", "mil gracias"]

# Precalcula los embeddings de las frases de referencia
saludos_emb = st_model.encode(SALUDOS, convert_to_tensor=True)
agradecimientos_emb = st_model.encode(AGRADECIMIENTOS, convert_to_tensor=True)

def es_saludo_agradecimiento(texto: str, threshold=0.6) -> bool:
    # Calcula el embedding del mensaje del usuario
    user_emb = st_model.encode([texto], convert_to_tensor=True)
    # Calcula similaridad con saludos y agradecimientos
    max_saludo = float(util.cos_sim(user_emb, saludos_emb).max())
    max_agradecimiento = float(util.cos_sim(user_emb, agradecimientos_emb).max())
    # Si alguna similaridad supera el umbral, es saludo o agradecimiento
    return max(max_saludo, max_agradecimiento) > threshold

### FIN MANEJO DE SALUDOS Y AGRADECIMIENTOS ### 

def extract_link_from_context(context: str) -> str:
    match = re.search(r'\[Ver documento legal completo\]\([^)]+\)', context)
    if match:
        return match.group(0)
    return ""

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if es_saludo_agradecimiento(request.prompt):
            context = ""
        else:
            context = retrieve_documents_with_link(request.prompt)

        #context = retrieve_documents_with_link(request.prompt)
        
        # IMPRIME EL CONTEXTO EN LA CONSOLA PARA DEBUG
        print("=== SYSTEM PROMPT PASADO AL MODELO ===")
        print(SYSTEM_PROMPT)
        print("=== FIN SYSTEM PROMPT ===\n")
        print("=== CONTEXTO LEGAL PASADO AL MODELO ===")
        print(context)
        print("=== FIN CONTEXTO ===\n")
        
        prompt = build_context_prompt(request.prompt, context)
        link_md = extract_link_from_context(context)

        # Streaming de la respuesta del modelo
        async def answer_generator() -> AsyncGenerator[bytes, None]:
            response_text = ""
            try:
                for chunk in llm.stream(prompt):
                    if hasattr(chunk, "content") and chunk.content:
                        response_text += chunk.content
                        yield chunk.content.encode("utf-8")

                # No agregues el link si la respuesta es "No tengo suficiente información..."
                NO_INFO_MSG = "No tengo suficiente información en el contexto proporcionado para responder a la pregunta."
                if (
                    link_md
                    and link_md not in response_text
                    and NO_INFO_MSG not in response_text
                ):
                    yield f"\n\n{link_md}".encode("utf-8")
                yield b""
            except Exception as e:
                logging.error(f"Error en la invocación del modelo: {e}", exc_info=True)
                yield f"Error: {str(e)}".encode("utf-8")

        return StreamingResponse(answer_generator(), media_type="text/plain")

    except Exception as e:
        logging.error(f"Error en el endpoint /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "API de chat con GPT-4o y RAG funcionando correctamente."}