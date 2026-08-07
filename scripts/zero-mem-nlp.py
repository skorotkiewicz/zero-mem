#!/usr/bin/env python3
"""Persistent spaCy + BGE-M3 worker. Protocol: one JSON request/response per line."""

import json
import os
import sys

import spacy
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

SPACY_MODEL = os.getenv("ZERO_MEM_SPACY_MODEL", "en_core_web_sm")
BGE_MODEL = os.getenv("ZERO_MEM_BGE_MODEL", "BAAI/bge-m3")

transformers_logging.disable_progress_bar()
nlp = spacy.load(SPACY_MODEL, disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
encoder = SentenceTransformer(BGE_MODEL, device=os.getenv("ZERO_MEM_DEVICE") or None)
entity_cache: dict[str, list[tuple[str, str]]] = {}
embedding_cache: dict[str, object] = {}
query_embedding_cache: dict[str, object] = {}


def entity_records_for(texts: list[str]) -> list[list[tuple[str, str]]]:
    missing = list(dict.fromkeys(text for text in texts if text not in entity_cache))
    for text, doc in zip(missing, nlp.pipe(missing, batch_size=64)):
        entity_cache[text] = [
            (ent.text.lower().strip(), ent.label_)
            for ent in doc.ents
            if ent.text.strip()
        ]
    return [entity_cache[text] for text in texts]


def entities_for(texts: list[str]) -> list[list[str]]:
    return [[entity for entity, _label in records] for records in entity_records_for(texts)]


def entity_types_for(texts: list[str]) -> list[dict[str, list[str]]]:
    result: list[dict[str, list[str]]] = []
    for records in entity_records_for(texts):
        types: dict[str, list[str]] = {}
        for entity, label in records:
            if label not in types.setdefault(entity, []):
                types[entity].append(label)
        result.append(types)
    return result


def embeddings_for(texts: list[str]) -> list[object]:
    missing = list(dict.fromkeys(text for text in texts if text not in embedding_cache))
    if missing:
        vectors = encoder.encode_document(missing, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        embedding_cache.update(zip(missing, vectors))
    return [embedding_cache[text] for text in texts]


def query_embeddings_for(texts: list[str]) -> list[object]:
    missing = list(dict.fromkeys(text for text in texts if text not in query_embedding_cache))
    if missing:
        vectors = encoder.encode_query(missing, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        query_embedding_cache.update(zip(missing, vectors))
    return [query_embedding_cache[text] for text in texts]


def aligned_entity_seeds(query_entities: list[str], observed_entities: list[str]) -> dict[str, float]:
    if not query_entities or not observed_entities:
        return {}
    observed = list(dict.fromkeys(observed_entities))
    observed_vectors = embeddings_for(observed)
    result: dict[str, float] = {}
    for _query_entity, query_vector in zip(query_entities, query_embeddings_for(query_entities)):
        similarities = [float(query_vector @ vector) for vector in observed_vectors]
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        if similarities[best_index] > 0:
            entity = observed[best_index]
            result[entity] = max(result.get(entity, 0.0), similarities[best_index])
    return result


for line in sys.stdin:
    request = {}
    try:
        request = json.loads(line)
        query = request["query"]
        texts = request["texts"]
        document_vectors = embeddings_for(texts)
        query_vector = query_embeddings_for([query])[0]
        dense_scores = [float(query_vector @ vector) for vector in document_vectors]
        spacy_entities = entities_for(texts)
        hinted_entities = request.get("entityHints", [[] for _text in texts])
        entities = [
            list(dict.fromkeys([*detected, *hints]))
            for detected, hints in zip(spacy_entities, hinted_entities)
        ]
        query_entities = list(dict.fromkeys([
            *entities_for([query])[0],
            *request.get("queryEntityHints", []),
        ]))
        adjacency_scores = [
            1.0 if index == 0 else float(document_vectors[index - 1] @ document_vectors[index])
            for index in range(len(document_vectors))
        ]
        response = {
            "id": request["id"],
            "entities": spacy_entities,
            "entityTypes": entity_types_for(texts),
            "queryEntities": query_entities,
            "denseScores": dense_scores,
            "entitySeedScores": aligned_entity_seeds(query_entities, [entity for values in entities for entity in values]),
            "adjacencyScores": adjacency_scores,
        }
    except Exception as error:
        response = {"id": request.get("id"), "error": str(error)}
    print(json.dumps(response, ensure_ascii=False), flush=True)
