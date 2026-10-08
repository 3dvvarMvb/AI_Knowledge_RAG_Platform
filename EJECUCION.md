# Guía de Ejecución Rápida: AI Knowledge & RAG Platform (NeuralDocs)

Bienvenido a la guía oficial de ejecución del **Proyecto 4: AI Knowledge & RAG Platform (NeuralDocs)**. Este proyecto es una plataforma moderna de **Generación Aumentada por Recuperación (RAG - Retrieval-Augmented Generation)** y búsqueda semántica construida con **FastAPI, base de datos vectorial Qdrant, capa de caché en Redis y una interfaz web interactiva estilo Perplexity / ChatGPT**.

---

## 1. Puesta en Marcha (1 Solo Comando)

Desde esta carpeta (`04-ai-knowledge-rag-platform/`) o desde la raíz del workspace, ejecuta:

```bash
docker compose up -d --build
```

Esto levantará automáticamente 3 contenedores orquestados con dependencias de salud (`service_healthy`):
1. **Qdrant Vector Database (v1.11.3)**: Motor de búsqueda vectorial de ultra alto rendimiento con indexación HNSW y métrica de Coseno (`puerto 6333` HTTP API, `puerto 6334` gRPC).
2. **Redis 7 Alpine**: Motor en memoria para caché de consultas semánticas y reducción de latencia (`puerto 6382`).
3. **RAG Backend & NeuralDocs UI (FastAPI / Python 3.12)**: API REST de ingesta, fragmentación (chunking), generación de vectores densos normalizados, síntesis de respuestas aumentadas y dashboard web (`puerto 8083`).

Para verificar el estado de los contenedores:
```bash
docker compose ps
```

---

## 2. Interfaz Web NeuralDocs en Vivo

Abre tu navegador web y entra a:
👉 **[http://localhost:8083](http://localhost:8083)** (o `http://localhost:8083/dashboard`)

### 🧠 ¿Qué puedes experimentar en la interfaz?
1. **Sembrado Instantáneo**:
   - Haz clic en el botón superior **"Sembrar Documentos"**.
   - Qdrant se poblará automáticamente con 5 especificaciones técnicas sobre arquitecturas de software (PostgreSQL ACID Ledger, Microservicios Distribuidos, WebSockets en Tiempo Real, Motores Vectoriales y Telemetría IoT).
2. **Búsqueda Semántica & RAG en Tiempo Real**:
   - Escribe cualquier pregunta en lenguaje natural en la barra de consulta, por ejemplo:
     - *"¿Cómo previene el ledger contable los deadlocks y race conditions?"*
     - *"¿Cómo funciona el Circuit Breaker y el patrón SAGA en microservicios?"*
     - *"¿Cómo se sincronizan las tareas colaborativas en WebSockets?"*
   - Pulsa **"Consultar RAG"** y observa cómo:
     - El motor recupera los fragmentos (chunks) más relevantes en el espacio vectorial de Qdrant.
     - Te muestra el porcentaje de similitud matemática (Coseno).
     - Te despliega las **citas técnicas verificables** y una respuesta sintetizada sin alucinaciones.
3. **Ingesta Dinámica de Tus Propios Documentos**:
   - Pulsa **"Ingestar Documento"**.
   - Ingresa un título y pega un texto o especificación técnica.
   - El sistema automáticamente fragmentará el texto en chunks semánticos, generará los vectores densos y los indexará en Qdrant. Inmediatamente podrás consultarlo en el chat.

---

## 3. Pruebas Automatizadas de Extremo a Extremo

Hemos incluido una suite de pruebas automatizadas que valida todo el flujo vectorial y la caché:

```bash
./scripts/test_rag.sh
```

El script ejecutará automáticamente:
- **Verificación de salud** de FastAPI, Qdrant y Redis.
- **Servicio del Dashboard** interactivo.
- **Sembrado de documentos** en Qdrant.
- **Búsqueda semántica vectorial** y verificación de citas devueltas.
- **Validación del caché en Redis** en consultas repetidas.
- **Ingesta en caliente de un nuevo documento técnico** y recuperación semántica inmediata.

---

## 4. Endpoints de la API (Probados con `curl`)

### A. Healthcheck del Ecosistema
```bash
curl -s http://localhost:8083/health | jq
```
*Respuesta:*
```json
{
  "status": "healthy",
  "service": "ai-knowledge-rag-api",
  "qdrant": "connected",
  "redis_cache": "connected",
  "indexed_documents": 5,
  "total_vectors_in_qdrant": 5
}
```

### B. Sembrar la Base de Conocimiento
```bash
curl -s -X POST http://localhost:8083/api/v1/seed | jq
```

### C. Listar Documentos Indexados
```bash
curl -s http://localhost:8083/api/v1/documents | jq
```

### D. Consulta Semántica RAG (Retrieval + Augmented Synthesis)
```bash
curl -s -X POST http://localhost:8083/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cómo garantiza el ledger la consistencia ACID?",
    "top_k": 3
  }' | jq
```

### E. Ingestar y Fragmentar un Nuevo Documento
```bash
curl -s -X POST http://localhost:8083/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Protocolo gRPC vs REST en Microservicios",
    "content": "gRPC utiliza HTTP/2 y Protocol Buffers binarios para serialización de datos de ultra baja latencia. Ofrece tipado estricto en tiempo de compilación y soporte nativo para streaming bidireccional.",
    "metadata": {
      "category": "Backend Protocols",
      "author": "Tech Lead"
    }
  }' | jq
```

---

## 5. Conceptos de Arquitectura para tus Entrevistas Técnicas

Cuando un reclutador o líder técnico te pregunte sobre este proyecto, destaca estos pilares de ingeniería:

1. **¿Qué problema resuelve RAG y por qué no usar solo un LLM estándar?**
   - *Respuesta*: Los LLMs sufren de alucinaciones y tienen conocimientos estáticos limitados por su fecha de corte de entrenamiento. La arquitectura RAG consulta primero una base vectorial privada indexada en tiempo real (Qdrant), extrae los fragmentos con mayor afinidad semántica y los suministra como contexto obligatorio, produciendo respuestas precisas, auditables y con citas verificables.
2. **¿Por qué es fundamental la fragmentación (Chunking)?**
   - *Respuesta*: Si vectorizas un documento de 50 páginas como un único vector, los conceptos específicos se diluyen en un promedio global difuso. Al fragmentarlo en chunks con sliding window, cada fragmento representa un concepto atómico con alta densidad semántica, lo que permite a la base vectorial encontrar la respuesta exacta en milisegundos.
3. **¿Por qué Qdrant y la métrica de Coseno?**
   - *Respuesta*: Qdrant es una base de datos vectorial de vanguardia escrita en Rust, altamente escalable y optimizada para el algoritmo HNSW (*Hierarchical Navigable Small World*). La similitud de Coseno calcula el ángulo entre dos vectores normalizados independientemente de su magnitud, lo que mide la orientación conceptual de la consulta con respecto al documento.
4. **¿Qué rol cumple Redis como capa de caché semántico?**
   - *Respuesta*: Las búsquedas vectoriales y los llamados a modelos de síntesis son computacionalmente costosos. Redis almacena en caché respuestas frecuentes con un TTL controlado, reduciendo la latencia de decenas de milisegundos a menos de 1 milisegundo y ahorrando costo computacional.
5. **Multi-Stage Docker & Non-Root Security**:
   - El `Dockerfile` utiliza un patrón multi-stage (builder + runtime con usuario `appuser` uid 1000) minimizando el tamaño de la imagen y garantizando seguridad en entornos cloud regulados.

---

## 6. Despliegue en Kubernetes (`k8s/`)

El proyecto incluye manifiestos completos en la carpeta `k8s/`:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/qdrant-statefulset.yaml
kubectl apply -f k8s/rag-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## 7. Detener el Entorno

Para detener y limpiar los contenedores sin borrar los datos indexados:
```bash
docker compose down
```

Para reiniciar desde cero eliminando los volúmenes de datos:
```bash
docker compose down -v
```
