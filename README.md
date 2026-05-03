# ARTROOM AI

## NAME

Jose Pajuelo

## DESCRIPTION

ArtRoom AI is an interactive React application backed by a Python FastAPI API. It combines:

1. A retrieval-augmented chatbot (RAG) with LangChain.
2. Text-to-image generation.
3. Image-to-image artistic transformation.
4. A custom fine-tuned diffusion model for Pancho Fierro.
5. Prompt enhancement using GPT.

## FEATURES

### RAG Chatbot

The chatbot answers as the selected artist in first person. It loads artist web pages, stores them in a FAISS index, retrieves relevant context, and answers only from that verified context.

If the answer is not found, it replies: "I would prefer not to talk about that."

### Text-to-Image Generation

Generates artwork from a user prompt and the selected artist style. Non-Pancho artists use Replicate model `black-forest-labs/flux-1.1-pro`.

### Image-to-Image Transformation

Uploads a JPG or PNG and applies the selected artist style while preserving the original image structure. Uses Replicate model `prunaai/p-image-edit`.

### Fine-Tuned Pancho Fierro Model

Pancho Fierro text-to-image requests use the fine-tuned Replicate model configured in `utils/sd_api.py`.

## INSTALLATION

### Windows

1. Install Python dependencies:

```powershell
uv sync
```

2. Install React dependencies:

```powershell
npm install
```

3. Set environment variables if they are not already in `.env`:

```powershell
$env:REPLICATE_API_TOKEN="your_token_here"
$env:OPENAI_API_KEY="your_key_here"
```

4. Start the API:

```powershell
uv run uvicorn api:app --reload
```

5. In a second terminal, start React:

```powershell
npm run dev
```

6. Open:

```text
http://127.0.0.1:5173
```

### Mac / Linux

1. Install Python dependencies:

```bash
uv sync
```

2. Install React dependencies:

```bash
npm install
```

3. Set environment variables if they are not already in `.env`:

```bash
export REPLICATE_API_TOKEN="your_token_here"
export OPENAI_API_KEY="your_key_here"
```

4. Start the API:

```bash
uv run uvicorn api:app --reload
```

5. In a second terminal, start React:

```bash
npm run dev
```

6. Open:

```text
http://127.0.0.1:5173
```

## PROJECT STRUCTURE

```text
ArtRoom_AI/
├── api.py
├── app.py
├── config/
│   └── styles.json
├── index.html
├── main.py
├── package.json
├── pyproject.toml
├── src/
│   ├── main.jsx
│   └── styles.css
├── utils/
│   ├── chatbot_rag.py
│   ├── prompt_generator.py
│   ├── sd_api.py
│   └── pictures/
└── vite.config.js
```

## ACADEMIC CONTEXT

This project was completed for the course MPCS 57200 Generative AI at the University of Chicago.
