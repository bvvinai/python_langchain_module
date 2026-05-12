# Python LangChain Module

Complete LangChain starter project with a single shared configuration for:
- LLM providers
- Embedding providers
- Vector databases

Current default stack in config:
- LLM: `ollama` (`llama3.2`)
- Embeddings: `ollama` (`nomic-embed-text`)
- Vector DB: `chroma`

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
python src/main.py
```

Common provider config is in `config/providers.yaml`.

Active providers are selected by env vars:
- `ACTIVE_LLM_PROVIDER`
- `ACTIVE_EMBEDDING_PROVIDER`
- `ACTIVE_VECTOR_DB_PROVIDER`

Supported providers:
- LLM: `openai`, `anthropic`, `google`, `ollama`
- Embeddings: `openai`, `huggingface`, `ollama`
- Vector DB: `chroma`, `faiss`, `qdrant`

OpenAI-compatible gateway support:
- You can point both OpenAI chat and embeddings to a custom gateway URL (for example a hackathon endpoint) using `OPENAI_BASE_URL`.
- Example: `OPENAI_BASE_URL=https://tcsopenai.com/v1`
- Precedence: env var `OPENAI_BASE_URL` first, then `base_url` from `config/providers.yaml`.

Anthropic custom endpoint support:
- You can set a custom Anthropic endpoint using `ANTHROPIC_API_URL` (preferred) or `ANTHROPIC_BASE_URL`.
- Example: `ANTHROPIC_API_URL=https://your-anthropic-gateway.example.com`
- Precedence: `ANTHROPIC_API_URL`, then `ANTHROPIC_BASE_URL`, then `base_url` from `config/providers.yaml`.

## Architecture Flowchart (All Functions + Inputs/Outputs)

```mermaid
flowchart TD
  %% Entry points
  U1["User or Script"] --> M1["main.main\nInput HOST PORT RELOAD env\nOutput uvicorn server process"]
  M1 --> A0["app.api.app create_app\nInput none\nOutput FastAPI app instance"]

  U2["CLI User"] --> C0["app.cli Typer commands"]

  %% API routes
  A0 --> A1["health endpoint\nInput none\nOutput status ok"]
  A0 --> A2["providers endpoint\nInput none\nOutput ProvidersResponse"]
  A0 --> A3["chat endpoint\nInput ChatRequest prompt\nOutput ChatResponse response"]
  A0 --> A4["index endpoint\nInput IndexRequest data_dir chunk_size chunk_overlap\nOutput IndexResponse chunks_indexed"]
  A0 --> A5["ask endpoint\nInput AskRequest question k\nOutput AskResponse answer"]
  A0 --> A6["upload endpoint\nInput multipart file plus data_dir\nOutput UploadResponse file metadata"]

  %% CLI commands
  C0 --> C1["show_providers\nInput none\nOutput console text"]
  C0 --> C2["chat command\nInput prompt\nOutput console text"]
  C0 --> C3["index_data\nInput data_dir chunk_size chunk_overlap\nOutput console text"]
  C0 --> C4["ask command\nInput question k\nOutput console text"]

  %% Shared config loading
  A2 --> L0
  A3 --> L0
  A4 --> L0
  A5 --> L0
  C1 --> L0
  C2 --> L0
  C3 --> L0
  C4 --> L0

  L0["config.loader load_app_config\nInput config_path or APP_CONFIG_PATH\nOutput ResolvedConfig"] --> L1["_read_yaml\nInput config_path Path\nOutput dict"]
  L0 --> L2["_active_provider\nInput env_key default\nOutput provider_name"]
  L0 --> CM0

  %% Config models
  CM0["AppConfig model_validate\nInput raw yaml dict\nOutput AppConfig"]
  CM0 --> CM1["DefaultsConfig\nInput llm embedding vector db provider names\nOutput typed defaults"]
  L0 --> CM2["ActiveProviders\nInput selected provider names\nOutput ActiveProviders"]
  L0 --> CM3["ResolvedConfig\nInput active plus section configs plus raw AppConfig\nOutput ResolvedConfig"]
  CM3 --> CM4["ResolvedConfig get\nInput section key default\nOutput selected value"]

  %% LLM factory path
  A3 --> F1
  A5 --> F1
  C2 --> F1
  C4 --> F1
  F1["build_llm\nInput ResolvedConfig\nOutput BaseChatModel"]
  F1 --> F1a["_optional_args\nInput llm config dict\nOutput args dict"]
  F1 --> F1b["_get_env\nInput env var name\nOutput value or none"]

  %% Embedding factory path
  A4 --> F2
  A5 --> F2
  C3 --> F2
  C4 --> F2
  F2["build_embeddings\nInput ResolvedConfig\nOutput Embeddings"]

  %% Ingestion path
  A4 --> I1
  C3 --> I1
  I1["load_documents\nInput data_dir\nOutput list of Document"] --> I2["split_documents\nInput documents chunk_size chunk_overlap\nOutput list of Document"]

  %% Vector store path
  I2 --> V1
  A5 --> V1
  C4 --> V1
  F2 --> V1
  V1["build_vector_store\nInput ResolvedConfig Embeddings optional documents\nOutput VectorStore"]

  %% RAG chain path
  A5 --> R1
  C4 --> R1
  F1 --> R1
  V1 --> R1
  R1["build_rag_chain\nInput BaseChatModel BaseRetriever\nOutput runnable chain answer text"] --> R2["_format_documents\nInput list of Document\nOutput context string"]

  %% API schema models
  S1["ChatRequest\nInput fields prompt\nOutput validated request model"]
  S2["ChatResponse\nInput fields response\nOutput validated response model"]
  S3["IndexRequest\nInput fields data_dir chunk_size chunk_overlap\nOutput validated request model"]
  S4["IndexResponse\nInput fields chunks_indexed\nOutput validated response model"]
  S5["AskRequest\nInput fields question k\nOutput validated request model"]
  S6["AskResponse\nInput fields answer\nOutput validated response model"]
  S7["ProvidersResponse\nInput fields llm_provider embedding_provider vector_db_provider\nOutput validated response model"]
  S8["UploadResponse\nInput fields file_name saved_path size_bytes\nOutput validated response model"]

  A3 --> S1 --> S2
  A4 --> S3 --> S4
  A5 --> S5 --> S6
  A2 --> S7
  A6 --> S8
```


Health check:

```bash
curl http://127.0.0.1:8000/health
```

PowerShell alternative:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Providers:

```bash
curl http://127.0.0.1:8000/providers
```

PowerShell alternative:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/providers
```

Chat:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"prompt\": \"What is LangChain?\"}"
```

PowerShell alternative:

```powershell
$body = @{ prompt = "What is LangChain?" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body $body
```

Index documents:

```bash
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d "{\"data_dir\": \"data\", \"chunk_size\": 1000, \"chunk_overlap\": 150}"
```

PowerShell alternative:

```powershell
$body = @{ data_dir = "data"; chunk_size = 1000; chunk_overlap = 150 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/index -Method Post -ContentType "application/json" -Body $body
```

Ask with RAG:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Summarize indexed docs\", \"k\": 4}"
```

Ask with MMR ranking:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Summarize indexed docs\", \"k\": 4, \"ranking_strategy\": \"mmr\", \"fetch_k\": 16, \"lambda_mult\": 0.5}"
```

PowerShell alternative:

```powershell
$body = @{ question = "Summarize indexed docs"; k = 4 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body $body
```

PowerShell with MMR ranking:

```powershell
$body = @{ question = "Summarize indexed docs"; k = 4; ranking_strategy = "mmr"; fetch_k = 16; lambda_mult = 0.5 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body $body
```

Upload file to data folder:

```bash
curl -X POST http://127.0.0.1:8000/upload -F "file=@data/sample.pdf" -F "data_dir=data"
```

PowerShell alternative:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/upload -Method Post -Form @{ file = Get-Item "data/sample.pdf"; data_dir = "data" }
```

Upload behavior:
- The file is saved to `data_dir` and indexed immediately.
- Response includes `indexed_chunks` for that uploaded file.

Supported file types for upload/index: `.txt`, `.md`, `.csv`, `.pdf`.

Interactive docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
