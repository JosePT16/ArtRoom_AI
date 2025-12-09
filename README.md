# ARTROOM AI

ArtRoom AI is an interactive Streamlit application that combines four AI capabilities:

1.  A retrieval-augmented chatbot (RAG).

2. Text-to-image generation,

3. Image-to-image artistic transformation,

4. A custom fine-tuned diffusion model


## FEATURES

### RAG Chatbot

This chatbot answers as if it were the selected artist, speaking in first person.
It retrieves information from online sources using RAG and responds only with verified context.

The system loads artist webpages, splits them into chunks, creates embeddings, and stores them in a FAISS index.
When the user asks a question, the chatbot searches for the most relevant text and generates an answer based only on that information.

If the answer is not found, it replies: “I would prefer not to talk about that.”

### Text-to-Image Generation

This section generates artwork from the artist selected using
natural-language prompts enhanced by GPT.
Uses Replicate API diffusion model of "black-forest-labs/flux-1.1-pro"

### Image-to-Image Transformation

This section allow to upload images to turn it into artistic variations
in the style of the artist selected.
Uses Replicate API diffusion model of "prunaai/p-image-edit"

### Text-to-Image with Fine-Tuned Model

This section generates artwork from the artist selected using
natural-language prompts enhanced by GPT.

For generating the image, it use a model fine-tuned with 30 images
of artist Pancho Fierro. The model was fine-tuned by the page Replicate
(https://replicate.com/)


## INSTALLATION


### For Windows

#### 1. Clone the repository
- git clone git@github.com:JosePT16/ArtRoom_AI.git

####  2. Install uv if not installed
- pip install uv

####  3. Set environment variables
- $env: REPLICATE_API_TOKEN=your_token_here
- $env: OPENAI_API_KEY=your_key_here

#### 4. Run the program
- uv sync
- uv run streamlit run .\app.py


### For MAC

#### 1. Clone the repository
- git clone https://github.com/yourusername/artroom-ai.
- cd artroom-ai


####  2. Install uv if not installed
- curl -LsSf https://astral.sh/uv/install.sh | sh

####  3. Set environment variables
- export REPLICATE_API_TOKEN="your_token_here"
- export OPENAI_API_KEY="your_key_here"

#### 4. Run the program
- uv sync
- uv run streamlit run app.py


## PROJECT STRUCTURE

ArtRoom_AI/

app.py
utils/ (helper scripts)
config/ json archives
main.py

## ACADEMIC CONTEXT

This project was completed for the course:
MPCS 52700 — Generative AI
University of Chicago

Demonstrated concepts include:

- Text-to-image and image-to-image diffusion
- Fine-tuning using LoRA
- Construction of a RAG pipeline
- Full interactive application design using Streamlit


