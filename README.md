# Project 4: AI Knowledge RAG Platform

Motor empresarial de recuperación aumentada por generación (RAG) y búsqueda semántica de documentos técnicos utilizando FastAPI y base de datos vectorial Qdrant.

## Arquitectura
- **API Backend**: Python 3.12 (FastAPI, AsyncIO)
- **Vector Database**: Qdrant (almacenamiento y búsqueda de similitud coseno)
- **Caché y Colas**: Redis 7
- **DevOps**: Docker Compose & Manifiestos de Kubernetes

## Ejecución Rápida
```bash
# 1. Iniciar servicios con Docker Compose
docker compose up -d --build

# 2. Ingestar un documento técnico en la base de conocimiento
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Arquitectura de Microservicios",
    "content": "Los microservicios se comunican asíncronamente mediante Kafka y aseguran consistencia transaccional con patrones Saga y bases de datos aisladas.",
    "metadata": {"author": "edu", "category": "engineering"}
  }'

# 3. Realizar una búsqueda semántica
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cómo se comunican los microservicios?",
    "top_k": 2
  }'
```
