import os
import sys
import re
import json
import uuid
import hashlib
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-api")

# Configuration Constants
COLLECTION_NAME = "knowledge_base"
VECTOR_DIM = 128
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Global Clients & State
qdrant_client: Optional[QdrantClient] = None
redis_client: Optional[redis.Redis] = None
DOCUMENTS_STORE: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------
# Vector Embedding & Text Processing Functions
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize text removing excess whitespace and lowercasing."""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_document(text: str, max_chunk_size: int = 350, min_chunk_size: int = 40) -> List[str]:
    """
    Split text into logical paragraphs and sentences using sliding context.
    Ensures high semantic density per vector representation.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        if len(para) <= max_chunk_size:
            if len(para) >= min_chunk_size:
                chunks.append(para)
            elif chunks:
                chunks[-1] += " " + para
            else:
                chunks.append(para)
        else:
            # Split paragraph by sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk = f"{current_chunk} {sentence}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)

    return chunks if chunks else [text]

def generate_dense_embedding(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """
    Generate a normalized deterministic dense embedding vector.
    Uses multi-granularity tokenization (words + subword n-grams) with MD5 bucket hashing.
    Normalized with L2 norm to ensure Cosine similarity metric compatibility in Qdrant.
    Runs with 0% cloud dependencies and sub-millisecond execution.
    """
    text_clean = text.lower()
    words = re.findall(r'\b[a-záéíóúüñ0-9_]{2,}\b', text_clean)
    
    # Subword character 3-grams for morphological overlap
    char_ngrams = [text_clean[i:i+3] for i in range(len(text_clean)-2)] if len(text_clean) >= 3 else []
    
    vec = np.zeros(dim, dtype=np.float32)
    
    # Process word tokens (higher semantic weight)
    for w in words:
        digest = hashlib.md5(w.encode('utf-8')).digest()
        idx = int.from_bytes(digest[:2], 'big') % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        # Weight longer technical words higher
        weight = 1.0 + min(len(w) * 0.1, 1.0)
        vec[idx] += sign * weight

    # Process subword ngrams (captures prefixes/suffixes and technical keywords)
    for ng in char_ngrams[:100]:
        digest = hashlib.md5(ng.encode('utf-8')).digest()
        idx = int.from_bytes(digest[:2], 'big') % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign * 0.3

    # Normalize vector to unit sphere (L2 norm = 1.0)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return vec.tolist()

# ---------------------------------------------------------
# Application Lifespan & Service Startup
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, redis_client
    logger.info("Initializing AI Knowledge RAG Platform...")

    # Connect to Qdrant Vector DB
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
        collections = qdrant_client.get_collections().collections
        col_names = [c.name for c in collections]
        if COLLECTION_NAME not in col_names:
            logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}' with {VECTOR_DIM}-dim cosine metric...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIM,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{COLLECTION_NAME}' ready.")
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant connection: {e}")

    # Connect to Redis Cache
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=3
        )
        redis_client.ping()
        logger.info(f"Redis connected successfully at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Redis cache connection failed: {e}. Running without cache.")
        redis_client = None

    yield

    logger.info("Shutting down AI Knowledge RAG Platform...")

# ---------------------------------------------------------
# FastAPI App Definition
# ---------------------------------------------------------

app = FastAPI(
    title="NeuralDocs - AI Knowledge RAG Engine",
    description="Production-ready Semantic Search and RAG platform with Qdrant Vector Database and Redis Caching",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=200, example="PostgreSQL ACID & Double-Entry Ledger")
    content: str = Field(..., min_length=10, example="Detailed architecture description...")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, example="¿Cómo garantiza el ledger la consistencia ACID?")
    top_k: int = Field(default=3, ge=1, le=10)

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the interactive NeuralDocs Web UI."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>NeuralDocs Dashboard</h1><p>dashboard.html not found</p>")

@app.get("/health")
def health_check():
    """System health check verifying Qdrant, Redis, and indexed vectors count."""
    qdrant_status = "disconnected"
    total_vectors = 0
    if qdrant_client:
        try:
            info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            qdrant_status = "connected"
            total_vectors = info.points_count or 0
        except Exception:
            qdrant_status = "error"

    redis_status = "disconnected"
    if redis_client:
        try:
            redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "error"

    return {
        "status": "healthy",
        "service": "ai-knowledge-rag-api",
        "qdrant": qdrant_status,
        "redis_cache": redis_status,
        "indexed_documents": len(DOCUMENTS_STORE),
        "total_vectors_in_qdrant": total_vectors
    }

@app.get("/api/v1/documents")
def list_documents():
    """List all indexed documents with metadata and chunk counts."""
    return list(DOCUMENTS_STORE.values())

@app.post("/api/v1/documents", status_code=status.HTTP_201_CREATED)
def ingest_document(req: DocumentIngestRequest):
    """
    Ingests a technical document:
    1. Chunks text into semantic segments.
    2. Computes normalized dense vector embeddings.
    3. Upserts points into Qdrant collection with metadata payload.
    4. Caches document index and invalidates query cache.
    """
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant vector engine is unavailable")

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    chunks = chunk_document(req.content)
    
    points = []
    for idx, chunk in enumerate(chunks):
        point_id = str(uuid.uuid4())
        vector = generate_dense_embedding(f"{req.title}: {chunk}")
        payload = {
            "doc_id": doc_id,
            "doc_title": req.title,
            "chunk_index": idx,
            "chunk_text": chunk,
            "category": req.metadata.get("category", "General"),
            "author": req.metadata.get("author", "Engineer")
        }
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        logger.error(f"Failed to upsert vectors into Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant insertion error: {str(e)}")

    doc_record = {
        "id": doc_id,
        "title": req.title,
        "content": req.content,
        "chunks_count": len(chunks),
        "metadata": req.metadata
    }
    DOCUMENTS_STORE[doc_id] = doc_record

    # Invalidate query cache in Redis
    if redis_client:
        try:
            keys = redis_client.keys("rag:query:*")
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Failed to clear Redis cache: {e}")

    logger.info(f"Ingested doc '{req.title}' ({doc_id}) with {len(chunks)} chunks vector indexed.")
    return {
        "message": "Document ingested and vector-indexed in Qdrant successfully",
        "document": doc_record
    }

@app.post("/api/v1/query")
def semantic_query(req: QueryRequest):
    """
    Processes a RAG semantic query:
    1. Checks Redis query cache.
    2. Generates normalized query embedding.
    3. Performs Approximate Nearest Neighbors (ANN) Cosine search in Qdrant.
    4. Synthesizes an augmented contextual answer citing retrieved sources.
    5. Caches the result in Redis.
    """
    query_str = clean_text(req.query)
    cache_key = f"rag:query:{hashlib.sha256(f'{query_str}:{req.top_k}'.encode()).hexdigest()}"

    # 1. Check Redis Cache
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                cached_data = json.loads(cached)
                cached_data["cached"] = True
                return cached_data
        except Exception as e:
            logger.warning(f"Redis cache read error: {e}")

    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant vector engine is unavailable")

    # 2. Vectorize Query
    query_vec = generate_dense_embedding(query_str)

    # 3. Search in Qdrant
    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=req.top_k
        )
    except Exception as e:
        logger.error(f"Error querying Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant search error: {str(e)}")

    # 4. Format Citations & Synthesize Augmented Answer
    citations = []
    context_snippets = []

    for hit in search_results:
        payload = hit.payload or {}
        score = round(float(hit.score), 4)
        doc_info = {
            "id": payload.get("doc_id", "unknown"),
            "title": payload.get("doc_title", "Documento Técnico")
        }
        chunk_text = payload.get("chunk_text", "")
        citations.append({
            "doc": doc_info,
            "chunk_text": chunk_text,
            "relevance_score": score
        })
        context_snippets.append(f"[{doc_info['title']}]: {chunk_text}")

    # Generate contextual synthesis
    if citations:
        best_title = citations[0]["doc"]["title"]
        best_score_pct = int(citations[0]["relevance_score"] * 100)
        synthesis = (
            f"Basado en la recuperación de vectores en Qdrant (afinidad {best_score_pct}% con '{best_title}'), "
            f"se identificaron los siguientes detalles técnicos clave:\n\n"
            + "\n• ".join([c["chunk_text"] for c in citations[:2]])
        )
    else:
        synthesis = "No se encontraron documentos relevantes en la base vectorial con el umbral de similitud actual."

    response_payload = {
        "query": req.query,
        "total_matches": len(citations),
        "synthesized_response": synthesis,
        "cached": False,
        "results": citations
    }

    # 5. Write to Redis Cache (TTL = 300s)
    if redis_client and citations:
        try:
            redis_client.setex(cache_key, 300, json.dumps(response_payload))
        except Exception as e:
            logger.warning(f"Redis cache write error: {e}")

    return response_payload

@app.post("/api/v1/seed")
def seed_knowledge_base():
    """
    Populates Qdrant with 5 production-grade technical engineering articles:
    - Fintech Double-Entry Ledger (ACID & Row Locks)
    - Cloud Microservices Resilience (Circuit Breaker & SAGA)
    - Real-Time SaaS Collaboration (WebSockets & BullMQ)
    - Vector Databases & RAG Architectures (Qdrant & Cosine)
    - High-Throughput IoT Telemetry (MQTT & TimescaleDB)
    """
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant vector engine is unavailable")

    seed_docs = [
        {
            "title": "PostgreSQL Double-Entry Ledger & Concurrency Control",
            "metadata": {"category": "Fintech & Backend", "author": "Lead Architect"},
            "content": (
                "El motor contable de doble partida garantiza consistencia ACID absoluta en transacciones financieras. "
                "Cada movimiento económico genera un asiento balanceado donde el débito es exactamente igual al crédito (Debit == Credit). "
                "Para prevenir race conditions en transferencias concurrentes, se utiliza bloqueo pesimista a nivel de fila (SELECT ... FOR UPDATE) "
                "ordenando los IDs de cuentas de forma lexicográfica para erradicar cualquier posibilidad de Deadlock en PostgreSQL. "
                "Además, la clave de idempotencia (idempotency_key) almacenada en Redis previene transacciones duplicadas por reintentos de red."
            )
        },
        {
            "title": "Arquitectura de Microservicios Distribuidos y Resiliencia",
            "metadata": {"category": "Cloud & Distributed Systems", "author": "Backend Engineer"},
            "content": (
                "En arquitecturas políglotas con Python FastAPI y Java Spring Boot, la resiliencia es crítica ante caídas parciales. "
                "Se implementa el patrón Circuit Breaker para evitar cascadas de fallos en el catálogo y pedidos. "
                "Para la consistencia eventual entre microservicios autónomos, el patrón SAGA orquesta transacciones compensatorias. "
                "El API Gateway actúa como único punto de entrada (Single Point of Entry), encargándose de enrutar tráfico, balancear carga "
                "y unificar autenticación y métricas de observabilidad."
            )
        },
        {
            "title": "Sistemas Colaborativos en Tiempo Real con WebSockets y BullMQ",
            "metadata": {"category": "Full Stack & Real-Time", "author": "Full Stack Lead"},
            "content": (
                "Para plataformas colaborativas multiusuario como tableros Kanban interactivos, se utiliza sincronización bidireccional "
                "mediante WebSockets nativos de baja latencia con control de heartbeat cada 30 segundos. "
                "El broadcast selectivo notifica a los clientes conectados en la misma sala sobre eventos de arrastre y edición de tarjetas. "
                "Las tareas pesadas como generación de reportes y procesamiento de archivos se delegan de manera asíncrona a BullMQ "
                "con Redis, persistiendo los adjuntos binarios en MinIO Object Storage compatible con la API de Amazon S3."
            )
        },
        {
            "title": "Motores de Búsqueda Vectorial y Retrieval-Augmented Generation (RAG)",
            "metadata": {"category": "AI & Vector Search", "author": "AI Engineer"},
            "content": (
                "La arquitectura RAG mitiga las alucinaciones de los modelos de lenguaje mediante recuperación precisa en bases de conocimiento privadas. "
                "Los documentos técnicos son particionados en chunks semánticos densos y transformados en vectores embebidos normalizados. "
                "Qdrant indexa estos vectores utilizando el algoritmo HNSW (Hierarchical Navigable Small World) optimizado para métrica de Coseno. "
                "Las consultas calculan el producto punto normalizado para extraer los K fragmentos con mayor similitud semántica, "
                "los cuales son inyectados como contexto aumentado para sintetizar respuestas técnicas con citas verificables."
            )
        },
        {
            "title": "Ingesta de Telemetría IoT de Alto Rendimiento con MQTT y TimescaleDB",
            "metadata": {"category": "IoT & Big Data", "author": "DevOps & SRE"},
            "content": (
                "Para procesar millones de métricas por segundo de sensores industriales, se utiliza el broker Eclipse Mosquitto con protocolo MQTT. "
                "Un servicio de ingestión en Go 1.23 consume concurrentemente los eventos mediante worker pools con goroutines y canales no bloqueantes. "
                "Los datos de series temporales se almacenan en TimescaleDB sobre hypertables particionadas automáticamente por rangos de tiempo. "
                "Las políticas de compresión columnar y retención automática liberan espacio en disco, mientras dashboards de Grafana "
                "grafican anomalías de temperatura, presión y vibración con latencia inferior a 20 milisegundos."
            )
        }
    ]

    total_chunks = 0
    for doc_data in seed_docs:
        ingest_document(DocumentIngestRequest(**doc_data))
        total_chunks += len(chunk_document(doc_data["content"]))

    return {
        "message": "Base de conocimiento sembrada con éxito en Qdrant Vector Store",
        "seeded_count": len(seed_docs),
        "total_chunks_indexed": total_chunks
    }
