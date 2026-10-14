# Document Q&A

Local Document Q&A app with a React + TypeScript frontend, FastAPI backend,
and the original CLI. Upload a document, index it in an OpenAI vector store,
and ask questions with the Responses API's hosted File Search. SQLite is deferred.

## Browser app

Complete the Python setup below and set your key in `backend/.env`.
Start the API from the repository root:

```sh
backend/.venv/bin/python -m pip install -r backend/requirements.txt
backend/.venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

In another terminal, use Node 22.12+ (or Node 20.19+):

```sh
cd frontend
nvm use # if using nvm; run nvm install first if Node 22 is missing
npm install
npm run dev
```

Open the Local URL printed by Vite, normally `http://127.0.0.1:5173`.
Add a PDF, DOCX, TXT, or Markdown file (up to 20 MB), wait for processing,
select it, and ask a question. Expand Sources beneath an answer for citations.
Saved documents survive reloads; displayed question history lasts only for the
current browser session, and questions are independent rather than conversational.
The development server proxies `/api` to FastAPI using
[Vite's proxy configuration](https://vite.dev/config/server-options#server-proxy).
Multipart uploads follow [FastAPI's file upload API](https://fastapi.tiangolo.com/tutorial/request-files/).
The browser never receives the API key. Uploaded local copies are temporary and
removed after processing; the hosted files remain until CLI cleanup.

The API and CLI share the same state file. Run one API worker and stop it before
using CLI upload or cleanup; the API serializes its own requests with a lock.
Re-upload documents indexed by an earlier CLI version once to mark them ready
in the browser list (the existing remote upload is reused).
This is a local development app without authentication. Keep both servers on
loopback. No hosted deployment or GitHub push is performed.

```sh
cd frontend
npm run build # TypeScript checking and production asset build
```

The production build alone does not serve the API; use the development setup
above for this proof of concept.

## Setup

Use Python 3.11 or newer (this repository was verified with Python 3.13).
Run from the repository root:

```sh
python3.13 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env` locally and replace the placeholder with your OpenAI API key.
Do not paste keys into commands, chat, logs, or Git. An existing environment
variable takes precedence over `.env`. The CLI never logs the key or raw API
errors. `.env` files, virtual environments, uploads, and CLI state are ignored;
the placeholder-only `.env.example` is tracked.

## Usage

```sh
python backend/qa.py --help
python backend/qa.py ask ./uploads/handbook.pdf "What is the vacation policy?"
python backend/qa.py ask ./uploads/handbook.pdf "How do I request time off?"
```

Provide your own document; `uploads/` is an ignored place to keep local files.
PDF, DOCX, Markdown, and text are examples of supported formats. Consult the
[official supported-file list](https://developers.openai.com/api/docs/guides/tools-file-search#supported-files)
for restrictions. Documents are uploaded to OpenAI; use files you intend to send.
An API key, network access, model access, and available API quota are required.
Model inference, File Search, and vector storage can incur charges; see
[OpenAI pricing](https://developers.openai.com/api/docs/pricing).

The default model is `gpt-4.1-mini`, configurable with `OPENAI_MODEL` or
`ask ... --model MODEL`. Output includes the answer and the API's file citations:
filename, file ID, and annotation text index. The index refers to answer text,
not a page number. If no citations are returned, the CLI explicitly says so.

## Decisions and limitations

- One persistent vector store and a SHA-256-to-file-ID map live in ignored
  `backend/.document-qa/state.json`. Identical contents reuse the upload even if
  renamed or moved; changed contents create a new file. Old versions remain
  until cleanup. Each question filters File Search to the selected content hash.
- Processing is polled for up to five minutes before answering. IDs are saved
  after resource creation, so ordinary ingestion or question failures can be
  retried without uploading again. A failed/cancelled ingestion needs cleanup
  and retry. A stale ID after manual remote deletion produces an API error;
  cleanup tolerates already-deleted resources.
- State writes use atomic replacement. This small CLI supports one process at
  a time and one OpenAI project per state file. Keep the state file until cleanup;
  losing it loses the resource inventory. A crash between remote creation and
  saving its ID can leave an untracked resource; remove that in the OpenAI dashboard.
- Hosted File Search handles parsing, chunking, embedding, and retrieval. No
  local embeddings or database are needed. The browser uses a small FastAPI adapter.
  `store=False` avoids storing the generated response as application state;
  uploaded files and the vector store persist until deleted.

Implementation checked against official OpenAI documentation:
[File Search](https://developers.openai.com/api/docs/guides/tools-file-search),
[Retrieval](https://developers.openai.com/api/docs/guides/retrieval), and
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

## Cleanup

```sh
python backend/qa.py cleanup
```

This explicitly deletes **all resources tracked by this CLI**: its vector store,
uploaded files, and state file. Deleting a store alone does not delete uploaded
Files, so the command deletes both. Partial cleanup saves progress and can be
rerun. Run with the same OpenAI project credentials used to upload. It leaves
your original local documents, `.env`, and virtual environment in place.
Do not delete local state as a substitute for remote cleanup. For untracked
resources, inspect Files and vector stores in the OpenAI dashboard.

## Local verification

```sh
python -m unittest discover -s backend/tests -v
python -m pip check
```

Offline tests use mocked SDK calls to check reuse, document scoping, polling,
citation display, retryable cleanup, HTTP upload validation, temporary-file removal,
and safe API errors. They do not call OpenAI. Frontend TypeScript checking and the
Vite production build were verified. Browser interaction testing was unavailable.
Live uploading,
indexing, answer quality, model access, and actual citations require an API key
and have not been verified locally. To verify live, ask twice about one document,
confirm the second run says `Reusing uploaded document`, inspect the citations,
then run cleanup. No changes have been pushed to GitHub.
