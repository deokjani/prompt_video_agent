import stanza
from transformers import pipeline
from konlpy.tag import Okt

# 초기화
stanza.download("ko")  # 최초 1회만 실행
nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse")
okt = Okt()

# 모델 로딩
ner_pipeline = pipeline("ner", model="klue/bert-base", aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
t2t_pipe = pipeline("text-generation", model="skt/kogpt2-base-v2")
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

# 조사 제거 함수
def clean_josa(word: str) -> str:
    josa_list = ["에서", "에게", "으로", "로", "는", "은", "가", "이", "를", "을", "에", "도", "만"]
    for josa in sorted(josa_list, key=len, reverse=True):
        if word.endswith(josa):
            return word[:-len(josa)]
    return word

# 주어 추출 (Stanza)
def extract_subject_stanza(text: str) -> str:
    doc = nlp(text)
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel == "nsubj":
                return word.text
    return ""

# 장소 추출 (NER + QA 백업)
def extract_location(text: str) -> str:
    ner_results = ner_pipeline(text)
    ner_places = [ent["word"] for ent in ner_results if ent["entity_group"] == "LOC"]

    qa_place = qa_pipeline({
        "question": "이 문장에서 장소는 어디인가요?",
        "context": text
    })["answer"].strip()

    print("장소 후보 (NER):", ner_places)
    print("장소 후보 (QA):", qa_place)

    return ner_places[0] if ner_places else qa_place

# 형용사 추출 (주어와 가까운 위치 기준)
def extract_adjectives(text: str, subject: str) -> list:
    tokens = okt.pos(text)
    subject_index = next((i for i, (w, _) in enumerate(tokens) if subject in w), -1)
    adjectives = [w for i, (w, pos) in enumerate(tokens) if pos == "Adjective" and i < subject_index + 3]
    return adjectives

# 동사 추출
def extract_main_verb(text: str) -> str:
    verbs = [w for w, pos in okt.pos(text, stem=True) if pos == "Verb"]
    return verbs[0] if verbs else "하다"

# 동사 프레이즈 생성
def rewrite_action_phrase(action: str) -> str:
    prompt = f"""동사를 자연스럽고 다양하게 묘사하는 예시:
    걷다 → 걷는 장면이 담긴  
    웃다 → 웃고 있는 모습이 담긴  
    요리하다 → 요리하는 장면을 담은  
    연주하다 → 악기를 연주하는 장면을 포착한  
    {action} →"""
    result = t2t_pipe(prompt, max_new_tokens=30)[0]["generated_text"]
    lines = result.strip().splitlines()
    for line in lines:
        if "→" in line:
            parts = line.split("→")
            if len(parts) > 1:
                candidate = parts[1].strip()
                if 3 < len(candidate) < 50:
                    return candidate
    return f"{action[:-1]}는 장면이 담긴" if action.endswith("다") else f"{action}하는 장면이 담긴"

# 감정 분류 (zero-shot)
def extract_best_entity(text: str, candidates: list) -> str:
    result = classifier(
        text,
        candidate_labels=candidates,
        hypothesis_template="이 문장은 {} 감정의 영상입니다."
    )
    return result["labels"][0].split(" ")[0] if result["labels"] else ""

# 최종 프롬프트 생성
def generate_prompt(user_input: str, style_hint: str = None) -> dict:
    print("=" * 50)
    print("사용자 입력:", user_input)

    subject_raw = extract_subject_stanza(user_input)
    subject = clean_josa(subject_raw)

    place = extract_location(user_input)
    place_clean = clean_josa(place)

    adjectives = extract_adjectives(user_input, subject)
    action = extract_main_verb(user_input)
    action_phrase = rewrite_action_phrase(action)

    mood_candidates = [
        "우울한 (슬픈 분위기)", "슬픈 (감정이 가라앉은)", "어두운 (무거운 느낌)",
        "밝은 (즐거운 느낌)", "잔잔한 (조용하고 편안한)", "따뜻한 (온화한 감성)",
        "몽환적인 (꿈같고 흐릿한)", "즐거운 (유쾌하고 활발한)"
    ]
    mood = extract_best_entity(user_input, mood_candidates)

    print("주어 (Stanza):", subject_raw)
    print("형용사 (Okt):", adjectives)
    print("동사 (Okt):", action)
    print("동사 프레이즈:", action_phrase)
    print("최종 선택된 장소:", place_clean)
    print("감정 (zero-shot):", mood)

    # 스타일 힌트와 감정 병합 처리
    mood_clause = f"{mood} 분위기의" if mood else ""
    style_clause = style_hint.strip() if style_hint else ""
    if "분위기" in style_clause:
        mood_clause = ""

    final_clause = f"{mood_clause} {style_clause} 10초 영상".strip()

    # 형용사 + 주어 조합
    subject_phrase = " ".join(adjectives + [subject]) if subject else "무언가"

    parts = []
    if subject_phrase:
        parts.append(f"{subject_phrase}가")
    if place_clean:
        parts.append(f"{place_clean}에서")
    if action_phrase:
        parts.append(action_phrase)
    parts.append(final_clause)

    final_prompt = " ".join(parts)
    print("최종 프롬프트:", final_prompt)
    print("=" * 50)

    return {
        "auto_prompt": final_prompt,
        "components": {
            "subject": subject_phrase,
            "place": place_clean,
            "action": action,
            "mood": mood,
            "style_hint": style_hint or ""
        }
    }
