from prompt_history_recommender import get_recommendation_prompt
from base_prompt_engine import generate_prompt

SIMILARITY_THRESHOLD = 0.75

def select_prompt(current_input: str):
    candidates = get_recommendation_prompt(current_input, top_n=1)

    if not candidates:
        return generate_prompt(current_input), "generated"

    top = candidates[0]
    if top["similarity_score"] >= SIMILARITY_THRESHOLD:
        return generate_prompt(current_input, style_hint=top.get("style")), "recommended"
    else:
        return generate_prompt(current_input), "generated"
