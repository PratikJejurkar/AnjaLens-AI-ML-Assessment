# Task 4 – Maintenance Assistant with RAG and Equipment Tools

## 1. Objective

Provide a REST API that answers maintenance questions using supplied Markdown manuals, retrieves relevant manual passages, generates an answer with a Hugging Face-hosted language model, cites retrieved source documents, and supports two simple equipment-related tools.

## 2. Technology

- FastAPI and Uvicorn for the HTTP API
- Pydantic for request/response validation
- LangChain document loaders and recursive text splitting
- Hugging Face sentence-transformer embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Chroma as the persistent vector store
- Hugging Face Endpoint with `mistralai/Mistral-7B-Instruct-v0.3`
- JSON file for mock inspection records
- python-dotenv for local environment configuration

## 3. Knowledge sources

The `data/` directory contains:

- `pump_manual.md`
- `conveyor_sop.md`
- `safety_guide.md`

At startup, the application loads Markdown files, splits them into 500-character chunks with 50-character overlap, creates embeddings and stores them in Chroma. It loads an existing local vector store if one is present at the configured persistence path.

## 4. Configuration

Create `Task_4/.env` locally:

```text
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
```

Never commit `.env` or expose the token in source code, logs, screenshots or documentation. The token is required for the configured hosted LLM. The application can initialize with no LLM, but `/ask` returns a configuration message rather than generated answers.

## 5. Run (Windows CMD)

From the repository root:

```bat
cd Task_4
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The root-level `requirements.txt` can also be used if all tasks are being installed together.

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## 6. API endpoints

### `GET /health`

Returns service status, `llm_connected`, and `vectorstore_loaded`.

### `POST /ask`

Request:

```json
{
  "question": "What is the maximum pressure for the Hydraulic Pump Model X-200?"
}
```

Response shape:

```json
{
  "answer": "The maximum pressure is 3000 PSI. Do not exceed this limit.",
  "citations": ["pump_manual.md"],
  "tool_calls": []
}
```

The exact generated answer can vary by model response.

## 7. Tool behavior

- `get_equipment_status(equipment_id)` returns mock status data. The current implementation derives status from the numeric part of the equipment ID (even → operational; odd → warning) and returns a fixed service date/alarm example. This is demonstration logic, not live equipment telemetry.
- `log_inspection(equipment_id, note, severity)` appends a timestamped record to `Task_4/inspections.json`. The current API detects tool requests using keyword checks and extracts an equipment ID matching letters-hyphen-digits plus a note enclosed in single quotes.

## 8. Limitations and next improvements

- Replace mock equipment status with an authenticated live equipment/CMMS/PLC integration.
- Replace JSON inspection persistence with a transactional database and audit trail.
- Use explicit tool schemas/function calling rather than keyword and regular-expression routing.
- Add authentication, authorization, rate limits, request IDs, structured logs and API tests.
- Add retrieval score thresholds, metadata filtering, document versioning and re-indexing controls.
- Add groundedness evaluation and safety rules for maintenance recommendations; require human confirmation for actions that can affect equipment.
- The current implementation returns no LLM-generated answer if the Hugging Face token is absent.
