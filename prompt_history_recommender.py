import json
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

qa_pipe = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
sbert_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_sbert(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = sbert_model(**inputs)
    embeddings = output[0]
    mask = inputs["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
    pooled = torch.sum(embeddings * mask, 1) / mask.sum(1)
    return pooled.cpu().numpy()

def extract_style(prompt_text: str) -> str:
    result = qa_pipe({
        "question": "이 문장의 분위기나 스타일을 한 문장으로 설명해줘.",
        "context": prompt_text
    })
    return result["answer"]

def get_recommendation_prompt(current_input: str, top_n: int = 1, threshold: float = 0.75):
    path = "data/prompt_history_log.json"
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    past_inputs = [item["input"] for item in history]
    if not past_inputs:
        return []

    embeddings = encode_sbert(past_inputs + [current_input])
    current_vec = embeddings[-1]
    past_vecs = embeddings[:-1]
    scores = cosine_similarity([current_vec], past_vecs)[0]

    results = []
    for idx in scores.argsort()[::-1][:top_n]:
        if scores[idx] >= threshold:
            item = history[idx]
            style = extract_style(item.get("auto_korean_prompt", ""))
            results.append({
                "similarity_score": float(scores[idx]),
                "style": style
            })
    return results
