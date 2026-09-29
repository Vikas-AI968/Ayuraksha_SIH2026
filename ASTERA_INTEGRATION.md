# Astera frontend integration

The original IP-SAKTI Sahayak backend remains in its existing architecture. The cinematic Astera UI has been added under `astera-frontend/` as a separate frontend application.

## Backend

Run the existing backend exactly as before from this `BABA/` directory.

The Astera chat connects to:

`POST /api/v1/query`

## Astera frontend

From `BABA/astera-frontend/`:

```bash
npm install
npm run dev
```

The UI defaults to `http://localhost:8000` for the FastAPI backend. To change it, copy `.env.example` to `.env` and set `VITE_API_BASE_URL`.

The existing `frontend/` directory shipped with the backend is intentionally left untouched.

## What is connected

- Ask Anything composer -> `/api/v1/query`
- Session IDs are generated and reused during a chat
- Backend answer is rendered in the Astera conversation UI
- Citation list is rendered from the backend response
- Confidence score is rendered when returned
- Backend errors are shown in the conversation instead of silently failing
- Browser speech recognition can populate the composer when supported

The RAG metric cards remain part of the Astera UI:

- Retrieval Hit Rate/Recall@k — `R`
- Citation Correctness — existing `≈` artwork
- Faithfulness — existing `✳` artwork
