import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering

model_fpath = "/mnt/data/fujiaqi/ckpt/princeton-nlp/sup-simcse-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_fpath)
model = AutoModel.from_pretrained(model_fpath).cuda()
max_seq_length = 256

def compute_emb(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_seq_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy()

def cluster(texts, d=0.1):
    if len(texts) < 2:
        return [0]
    embs = compute_emb(texts)
    clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=d).fit(embs)
    return clustering.labels_.tolist()


def similarity(query, candidates):
    if not candidates:
        return []
    embs = compute_emb([query, *candidates])
    query_emb = embs[0]
    candidate_embs = embs[1:]
    return np.dot(candidate_embs, query_emb).tolist()

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]
    d: float

class OutputPrediction(BaseModel):
    labels: List


class SimilarityInput(BaseModel):
    query: str
    candidates: List[str]


class SimilarityOutput(BaseModel):
    similarities: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    labels = cluster(input_text.texts, input_text.d)
    return {"labels": labels}


@app.post("/similarity", response_model=SimilarityOutput)
async def predict_similarity(input_text: SimilarityInput):
    similarities = similarity(input_text.query, input_text.candidates)
    return {"similarities": similarities}
