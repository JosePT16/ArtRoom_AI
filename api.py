import json
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from utils.chatbot_rag import ARTIST_URLS, get_artist_answer
from utils.prompt_generator import build_prompt, build_prompt_image, build_prompt_pancho
from utils.sd_api import generate_image_pancho, generate_image_sd, generate_style_image


BASE_DIR = Path(__file__).resolve().parent
PICTURES_DIR = BASE_DIR / "utils" / "pictures"
HERO_DIR = BASE_DIR / "utils" / "hero"
ICONS_DIR = BASE_DIR / "utils" / "icons"

ARTIST_TO_STYLE_KEY = {
    "Van Gogh": "van_gogh",
    "Monet": "monet",
    "Picasso": "picasso",
    "Velasquez": "velazquez",
    "Dali": "dali",
    "Pancho Fierro": "fierro-no-fine-tunned",
}

ARTIST_TO_IMAGE = {
    "Van Gogh": "Van Gogh.jpg",
    "Monet": "Monet.jpg",
    "Picasso": "Picasso.jpg",
    "Velasquez": "Velazquez.jpg",
    "Dali": "Dali.jpg",
    "Pancho Fierro": "Pancho Fierro.jpg",
}

with open(BASE_DIR / "config" / "styles.json", "r", encoding="utf-8") as f:
    styles = json.load(f)

app = FastAPI(title="ArtRoom AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if PICTURES_DIR.exists():
    app.mount("/pictures", StaticFiles(directory=PICTURES_DIR), name="pictures")

if HERO_DIR.exists():
    app.mount("/hero", StaticFiles(directory=HERO_DIR), name="hero")

if ICONS_DIR.exists():
    app.mount("/icons", StaticFiles(directory=ICONS_DIR), name="icons")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    artist: str
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


class TextToImageRequest(BaseModel):
    artist: str
    description: str


def get_style_key(artist: str) -> str:
    try:
        return ARTIST_TO_STYLE_KEY[artist]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail="Unknown artist.") from exc


def get_artist_image_url(artist: str) -> str:
    image_name = ARTIST_TO_IMAGE[artist]
    image_path = PICTURES_DIR / image_name
    version = image_path.stat().st_mtime_ns if image_path.exists() else 0
    return f"/pictures/{quote(image_name)}?v={version}"


@app.get("/api/artists")
def list_artists():
    return [
        {
            "name": artist,
            "displayName": styles[ARTIST_TO_STYLE_KEY[artist]]["name"],
            "styleKey": ARTIST_TO_STYLE_KEY[artist],
            "imageUrl": get_artist_image_url(artist),
        }
        for artist in ARTIST_URLS.keys()
    ]


@app.get("/api/hero-images")
def list_hero_images():
    if not HERO_DIR.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        {"name": image.name, "imageUrl": f"/hero/{quote(image.name)}"}
        for image in sorted(HERO_DIR.iterdir())
        if image.is_file() and image.suffix.lower() in image_extensions
    ]


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        return get_artist_answer(
            request.artist,
            request.question,
            [message.model_dump() for message in request.history],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/text-to-image")
def text_to_image(request: TextToImageRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")

    style_key = get_style_key(request.artist)

    try:
        if request.artist == "Pancho Fierro":
            final_prompt = build_prompt_pancho(request.description, "Pancho Fierro style")
            image_url = generate_image_pancho(final_prompt)
        else:
            style_prompt = styles[style_key]["style_prompt"]
            final_prompt = build_prompt(request.description, style_prompt)
            image_url = generate_image_sd(final_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"imageUrl": image_url, "prompt": final_prompt}


@app.post("/api/image-to-image")
async def image_to_image(
    artist: str = Form(...),
    strength: float = Form(0.65),
    image: UploadFile = File(...),
):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")

    style_key = get_style_key(artist)
    style_prompt = styles[style_key]["style_prompt"]

    try:
        init_image_bytes = await image.read()
        final_prompt = build_prompt_image(style_prompt)
        result_url = generate_style_image(
            init_image_bytes=init_image_bytes,
            style_prompt=final_prompt,
            strength=strength,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"imageUrl": result_url, "prompt": final_prompt}
