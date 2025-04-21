import stanza
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from konlpy.tag import Okt

# 형태소 분석기 및 NLP 파이프라인 초기화
nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse", verbose=False)
okt = Okt()

# 모델 로딩
ner_pipeline = pipeline("ner", model="klue/bert-base", aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

tokenizer_kogpt = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model_kogpt = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")


def clean_josa(word: str) -> str:
    josa_list = ["에서", "에게", "으로", "로", "는", "은", "가", "이", "를", "을", "에", "도", "만"]
    for josa in sorted(josa_list, key=len, reverse=True):
        if word.endswith(josa):
            return word[:-len(josa)]
    return word


def extract_subjects(text: str) -> list:
    doc = nlp(text)
    return [w.text for sent in doc.sentences for w in sent.words if w.deprel == "nsubj"]


def extract_location(text: str) -> str:
    ner_places = [ent["word"] for ent in ner_pipeline(text) if ent["entity_group"] == "LOC"]
    qa_place = qa_pipeline({
        "question": "이 문장에서 장소는 어디인가요?",
        "context": text
    })["answer"].strip()
    print("장소 후보 (NER):", ner_places)
    print("장소 후보 (QA):", qa_place)
    return clean_josa(ner_places[0]) if ner_places else clean_josa(qa_place)


def extract_adjectives(text: str, subjects: list) -> dict:
    tokens = okt.pos(text)
    result = {subj: [] for subj in subjects}
    for i, (w, pos) in enumerate(tokens):
        if pos == "Adjective":
            for subj in subjects:
                if subj in [t[0] for t in tokens[i+1:i+3]]:
                    result[subj].append(w)
    return result


def extract_verbs(text: str) -> list:
    return [w for w, pos in okt.pos(text, stem=True) if pos == "Verb"]


def rewrite_action_phrase_kogpt2(action: str) -> str:
    prompt = f"""다음 동작을 자연스럽게 묘사하는 장면 문장을 생성하세요:
    걷다 → 공원을 걷는 장면이 담긴
    웃다 → 환하게 웃고 있는 모습이 담긴
    요리하다 → 음식을 요리하는 장면이 담긴
    연주하다 → 악기를 연주하는 장면이 담긴
    {action} →"""

    inputs = tokenizer_kogpt(prompt, return_tensors="pt")
    outputs = model_kogpt.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer_kogpt.eos_token_id)
    decoded = tokenizer_kogpt.decode(outputs[0], skip_special_tokens=True)

    if f"{action} →" in decoded:
        generated = decoded.split(f"{action} →", 1)[-1].strip()
        return generated.split("\n")[0].strip()
    return f"{action}하는 장면이 담긴"


def extract_mood(text: str) -> str:
    mood_candidates = ["우울한", "슬픈", "어두운", "밝은", "잔잔한", "따뜻한", "몽환적인", "즐거운"]
    result = classifier(
        text,
        candidate_labels=mood_candidates,
        hypothesis_template="이 문장은 {} 분위기의 영상입니다."
    )
    return result["labels"][0] if result["labels"] else ""


def generate_prompt(user_input: str, style_hint: str = None) -> dict:
    print("=" * 60)
    print("사용자 입력:", user_input)

    subjects_raw = extract_subjects(user_input)
    subjects = [clean_josa(s) for s in subjects_raw]
    adjectives_map = extract_adjectives(user_input, subjects)
    verbs = extract_verbs(user_input)
    place = extract_location(user_input)
    mood = extract_mood(user_input)

    print("주어 (Stanza):", subjects_raw)
    print("형용사 (Okt):", adjectives_map)
    print("동사 (Okt):", verbs)
    print("최종 선택된 장소:", place)
    print("감정 (zero-shot):", mood)

    prompt_lines = []
    for i, subject in enumerate(subjects):
        adj = " ".join(adjectives_map.get(subject, []))
        full_subject = f"{adj} {subject}".strip()
        verb = verbs[i] if i < len(verbs) else "하다"
        action_phrase = rewrite_action_phrase_kogpt2(verb)
        print(f"동사 프레이즈 for {verb}:", action_phrase)

        line = f"{full_subject}가 {place}에서 {action_phrase}"
        prompt_lines.append(line)

    body = " 그리고 ".join(prompt_lines)
    final_prompt = f"{body} {mood} 분위기의 10초 영상".strip()
    final_prompt = final_prompt.replace("  ", " ")

    print("최종 프롬프트:", final_prompt)
    print("=" * 60)

    return {
        "auto_prompt": final_prompt,
        "components": {
            "subjects": subjects,
            "place": place,
            "verbs": verbs,
            "mood": mood,
            "style_hint": style_hint or ""
        }
    }
