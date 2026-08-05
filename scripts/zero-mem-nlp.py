#!/usr/bin/env python3
"""Persistent spaCy + BGE-M3 worker. Protocol: one JSON request/response per line."""

import json
import os
import sys

import spacy
from sentence_transformers import SentenceTransformer

SPACY_MODEL = os.getenv("ZERO_MEM_SPACY_MODEL", "en_core_web_sm")
BGE_MODEL = os.getenv("ZERO_MEM_BGE_MODEL", "BAAI/bge-m3")

nlp = spacy.load(SPACY_MODEL, disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
encoder = SentenceTransformer(BGE_MODEL, device=os.getenv("ZERO_MEM_DEVICE") or None)
entity_cache: dict[str, list[str]] = {}
embedding_cache: dict[str, object] = {}


def entities_for(texts: list[str]) -> list[list[str]]:
    missing = list(dict.fromkeys(text for text in texts if text not in entity_cache))
    for text, doc in zip(missing, nlp.pipe(missing, batch_size=64)):
        entity_cache[text] = list(dict.fromkeys(ent.text.lower().strip() for ent in doc.ents if ent.text.strip()))
    return [entity_cache[text] for text in texts]


def embeddings_for(texts: list[str]) -> list[object]:
    missing = list(dict.fromkeys(text for text in texts if text not in embedding_cache))
    if missing:
        vectors = encoder.encode_document(missing, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        embedding_cache.update(zip(missing, vectors))
    return [embedding_cache[text] for text in texts]


for line in sys.stdin:
    request = {}
    try:
        request = json.loads(line)
        query = request["query"]
        texts = request["texts"]
        query_vector = encoder.encode_query(query, normalize_embeddings=True, show_progress_bar=False)
        dense_scores = [float(query_vector @ vector) for vector in embeddings_for(texts)]
        response = {
            "id": request["id"],
            "entities": entities_for(texts),
            "queryEntities": entities_for([query])[0],
            "denseScores": dense_scores,
        }
    except Exception as error:
        response = {"id": request.get("id"), "error": str(error)}
    print(json.dumps(response, ensure_ascii=False), flush=True)
