import json
import os
from datetime import datetime

HISTORY_PATH = "data/prompt_history_log.json"

def save_prompt_history(entry: dict):
    entry["timestamp"] = str(datetime.now())
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
