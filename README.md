# Cisco Nexus 9000 Troubleshooting Assistant

A small, local RAG demo that retrieves passages from the supplied Cisco Nexus
9000 troubleshooting PDF and answers questions with Groq GPT-OSS 120B. The
PDF is embedded locally with `all-MiniLM-L6-v2` and indexed in persistent
ChromaDB; questions use similarity retrieval with the top five chunks.

## Requirements

- Windows with Python 3.12 (`py -3.12`)
- A Groq API key with access to `openai/gpt-oss-120b`
- Internet access for the first embedding-model download and Groq responses

The Cisco troubleshooting PDF is intentionally not included in this repository.
Place a licensed copy beside `app.py` with this filename:
`cisco-nexus-9000-series-nx-os-troubleshooting-guide-106x.pdf`.
The app reads the PDF in place and retains one-based page citations.

## Setup and run (PowerShell)

Run these commands from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in the existing `.env` file. Do not replace the file with
`.env.example`; that file only documents the expected variable.

Start the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860`).
Embeddings run locally on CPU. The first run downloads the embedding model,
extracts and splits the PDF (512-character chunks with 60-character overlap),
and creates `chroma_db/`. Later runs reuse the stored index.

To rebuild the index after replacing or updating the PDF, stop the app and
remove only the project-local `chroma_db` directory, then start the app again:

```powershell
Remove-Item -Recurse -Force .\chroma_db
.\.venv\Scripts\python.exe app.py
```

## Tests

Run the local unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the two live Groq smoke questions (requires the PDF, a configured API key,
and network access):

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

The smoke test asks a factual question about fan-module troubleshooting and a
broader question about investigating recurring switch issues. Answers include
the retrieved PDF page references where available.

## Troubleshooting

- **Missing PDF:** Place the supplied troubleshooting PDF beside `app.py`.
- **Missing API key:** Add `GROQ_API_KEY=...` to the existing `.env` file.
- **Model/API error:** Confirm the key is active, the account can access
  `openai/gpt-oss-120b`, and the machine can reach Groq.
- **No extractable text:** Scanned/image-only PDFs need OCR; OCR is not included.
- **Embedding download error:** Check internet access and retry; embeddings are
  downloaded on first use and then cached locally.
