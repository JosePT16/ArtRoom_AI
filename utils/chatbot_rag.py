import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_CACHE_DIR = BASE_DIR / ".rag_cache"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CACHE_VERSION = "1"


ARTIST_URLS = {
    "Van Gogh": [
        "https://en.wikipedia.org/wiki/Vincent_van_Gogh",
        "https://www.vangoghmuseum.nl/en/stories"
    ],
    "Monet": [
        "https://en.wikipedia.org/wiki/Claude_Monet",
    ],

    "Picasso": [
        "https://en.wikipedia.org/wiki/Pablo_Picasso",
    ],

    "Velasquez": [
        "https://en.wikipedia.org/wiki/Diego_Vel%C3%A1zquez%22",
    ],

    "Dali": [
        "https://en.wikipedia.org/wiki/Salvador_Dal%C3%AD",
    ],

    "Pancho Fierro": [
        "https://es.wikipedia.org/wiki/Pancho_Fierro",
    ]
}


def artist_cache_dir(artist_name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", artist_name.lower()).strip("-")
    return VECTOR_CACHE_DIR / slug


def cache_metadata(artist_name: str) -> dict:
    return {
        "artist": artist_name,
        "urls": ARTIST_URLS[artist_name],
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "cache_version": CACHE_VERSION,
    }


def is_cache_current(cache_dir: Path, expected_metadata: dict) -> bool:
    metadata_path = cache_dir / "metadata.json"
    index_path = cache_dir / "index.faiss"
    pickle_path = cache_dir / "index.pkl"

    if not metadata_path.exists() or not index_path.exists() or not pickle_path.exists():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            stored_metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return stored_metadata == expected_metadata


def save_cache_metadata(cache_dir: Path, metadata: dict) -> None:
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)



@lru_cache(maxsize=None)
def load_artist_embeddings(artist_name: str):
    """Loads a persisted FAISS index, or builds and saves it if missing."""

    urls = ARTIST_URLS[artist_name]
    embeddings = OpenAIEmbeddings()
    cache_dir = artist_cache_dir(artist_name)
    expected_metadata = cache_metadata(artist_name)

    if is_cache_current(cache_dir, expected_metadata):
        try:
            return FAISS.load_local(
                str(cache_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            # If local cache files are corrupted or incompatible, rebuild below.
            pass

    loader = WebBaseLoader(urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(cache_dir))
    save_cache_metadata(cache_dir, expected_metadata)

    return vectorstore



def format_history(history: list[dict], max_messages: int = 8) -> str:
    clean_messages = []
    for message in history[-max_messages:]:
        role = "Artist" if message.get("role") == "assistant" else "User"
        content = str(message.get("content", "")).strip()
        if content:
            clean_messages.append(f"{role}: {content}")

    return "\n".join(clean_messages) if clean_messages else "No previous messages."


def is_followup_question(question: str) -> bool:
    normalized = question.lower().strip()
    followup_terms = {
        "why",
        "how",
        "when",
        "where",
        "who",
        "what",
        "that",
        "this",
        "he",
        "she",
        "his",
        "her",
        "they",
        "them",
    }
    words = re.findall(r"[a-z']+", normalized)
    return len(words) <= 6 or any(word in followup_terms for word in words)


def build_retrieval_query(artist_name: str, question: str, history: list[dict]) -> str:
    if not is_followup_question(question):
        return f"{artist_name}: {question}"

    recent_history = format_history(history, max_messages=4)
    return f"{artist_name}\nRecent conversation:\n{recent_history}\nUser follow-up question: {question}"


def build_chain():
    """Creates a RunnableSequence LCEL RAG chain."""

    prompt = PromptTemplate(
        template="""
        The user is going to ask you questions as if you were the selected artist.
        Answer as {artist} in first person.
        Use ONLY the provided context to answer the question.
        Use the conversation history only to understand references like "he", "that", or "his brother".
        Do not invent facts from the conversation history.
        If the context does not directly answer the question but includes related artistic
        facts, answer carefully from those facts and say what you cannot know for sure.
        If the question asks for programming, coding, math homework, school assignments,
        recipes, travel planning, general advice, or any other topic unrelated to the
        artist, art, biography, or the conversation, politely decline in character
        without answering the unrelated request and without mentioning retrieved
        information, context, documents, databases, or sources.
        Decline unrelated questions by saying:
        "I would rather keep our conversation focused on my life and art."

        Conversation history:
        {history}

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["artist", "history", "context", "question"],
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    parser = StrOutputParser()

    # LCEL pipeline:
    chain = RunnableSequence(prompt | llm | parser)

    return chain



def get_artist_answer(artist_name: str, question: str, history: list[dict] | None = None):
    """
    Retrieves the most relevant chunks and runs LCEL chain.
    """

    history = history or []

    # Load FAISS embedding index for this artist
    docembeddings = load_artist_embeddings(artist_name)

    # Retrieve top chunks
    retrieval_query = build_retrieval_query(artist_name, question, history)
    results = docembeddings.similarity_search_with_score(retrieval_query, k=8)
    docs = [d[0] for d in results]

    # Combine chunks into RAG context
    context_text = "\n\n".join(doc.page_content for doc in docs)
    if not context_text.strip():
        return {
            "Answer": "I would prefer not to talk about that",
            "Reference": "",
            "Sources": [],
        }

    # Build chain
    chain = build_chain()

    # Invoke LCEL chain
    answer = chain.invoke({
        "artist": artist_name,
        "history": format_history(history),
        "context": context_text,
        "question": question
    })

    # Extract sources
    sources = []
    for doc in docs:
        if "source" in doc.metadata:
            sources.append(doc.metadata["source"])
        elif "url" in doc.metadata:
            sources.append(doc.metadata["url"])
    sources = list(set(sources))

    return {
        "Answer": answer,
        "Reference": context_text,
        "Sources": sources
    }
