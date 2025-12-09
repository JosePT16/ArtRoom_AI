import os
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence



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



@st.cache_resource
def load_artist_embeddings(artist_name: str):
    """Loads webpages, splits into chunks, embeds, and builds FAISS index."""
    
    urls = ARTIST_URLS[artist_name]

    loader = WebBaseLoader(urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore



def build_chain():
    """Creates a RunnableSequence LCEL RAG chain."""

    prompt = PromptTemplate(
        template="""
        The user is going to ask you questions as you were the artist
        Answer as you were the artist selected, talk in first person-
        Use ONLY the provided context to answer the question.
        If the answer is not in the context, say:
        "I would prefer not to talk about that"

        Context:
        {context}

        Question: {question}

        Answer:
        """,
                input_variables=["context", "question"]
            )

    llm = OpenAI(temperature=0)
    parser = StrOutputParser()

    # LCEL pipeline:
    chain = RunnableSequence(prompt | llm | parser)

    return chain



def get_artist_answer(artist_name: str, question: str):
    """
    Retrieves the most relevant chunks and runs LCEL chain.
    """

    # Load FAISS embedding index for this artist
    docembeddings = load_artist_embeddings(artist_name)

    # Retrieve top chunks
    results = docembeddings.similarity_search_with_score(question, k=2)
    docs = [d[0] for d in results]

    # Combine chunks into RAG context
    context_text = "\n\n".join(doc.page_content for doc in docs)

    # Build chain
    chain = build_chain()

    # Invoke LCEL chain
    answer = chain.invoke({
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
