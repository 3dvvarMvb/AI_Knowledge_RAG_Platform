# Project Directives: AI Knowledge & RAG Platform

Plataforma inteligente de Retrieval-Augmented Generation (RAG) con búsqueda semántica vectorial, ingestión de documentos y generación aumentada por contexto para LLMs.

## Responsabilidades de los Subagentes en este Proyecto
- **system-architect**: Diseñar la estrategia de chunking de texto (sliding window / semantic splitting) y esquemas de embeddings y metadatos vectoriales.
- **backend-engineer**: Desarrollar con FastAPI (`backend/`), integrando clientes de Qdrant o pgvector y pipelines de consulta de embeddings.
- **frontend-engineer**: Implementar en Next.js (`frontend/`) una interfaz de chat con streaming de tokens (SSE) y citas de fuentes.
- **devops-sre-engineer**: Configurar volúmenes para el vector store, variables de entorno para APIs de modelos y manifiestos de Kubernetes en `k8s/`.
- **qa-security-auditor**: Auditar la protección contra inyección de prompts (Prompt Injection) y sanitización de documentos subidos.
