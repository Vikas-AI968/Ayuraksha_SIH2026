# Astera + IP-SAKTI Sahayak Frontend

The Astera cinematic frontend is connected to the existing FastAPI IP-SAKTI Sahayak backend without changing the backend architecture or backend source files.

## Run

1. Start the backend from `BABA/` using the existing project instructions.
2. Copy `.env.example` to `.env` in this frontend folder if you need a custom backend URL.
3. Install and run the frontend:

```bash
npm install
npm run dev
```

The default API target is `http://localhost:8000`.

## Connected endpoint

The Ask Anything composer sends:

`POST /api/v1/query`

with `query`, `session_id`, `language`, `jurisdiction`, and `top_k`.

The UI renders the returned answer, confidence, status, and citations. Backend files and backend data are kept in their original `BABA/` project structure.
