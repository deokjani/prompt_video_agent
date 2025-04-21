# Prompt Video Agent

`Prompt Video Agent`는 자연어 문장을 기반으로 영상 프롬프트를 생성하고, 수정 이력을 저장하며, 과거 입력 히스토리를 참조하여 최적화된 프롬프트를 추천하는 시스템입니다.  
Streamlit 기반의 인터페이스를 통해 입력 → 프롬프트 생성 → 수정 → 결과 확인의 흐름을 직관적으로 제공합니다.

---

## 구성 및 주요 기능

### 1. 자연어 입력 해석
- 사용자가 입력한 문장에서 다음 요소를 추출:
  - **주어**: Stanza의 의존 구문 분석(`nsubj`)
  - **장소**: NER(`klue/bert-base`) → 실패 시 QA(`xlm-roberta-base-squad2`)
  - **동사**: Okt의 형태소 분석을 통해 추출
  - **형용사**: Okt 기반으로 주어 인접한 형용사 추출
  - **감정**: Zero-shot classification(`joeddav/xlm-roberta-large-xnli`)을 통해 문장의 분위기 해석

### 2. 프롬프트 생성
- 구성 요소들을 다음과 같은 문장 형식으로 결합:
  ```
  [형용사] [주어]가 [장소]에서 [동사 프레이즈] [감정] 분위기의 10초 영상
  ```
- **동사 프레이즈**: KoGPT2를 이용해 자연스러운 장면 묘사로 확장

### 3. 수정 및 이력 저장
- 사용자가 프롬프트를 수정하면 `prompt_diff.py`에서 변경 이력을 `difflib`로 비교
- 수정 전/후 내용은 `history_manager.py`를 통해 JSON 파일에 저장 (`prompt_history_log.json`)

### 4. 과거 기록 기반 추천
- `prompt_history_recommender.py`에서 SBERT(`sentence-transformers`) 임베딩을 통해 입력과 가장 유사한 과거 기록을 찾음
- 유사도가 임계값을 넘으면 해당 기록의 스타일 힌트를 추출하여 현재 입력에 반영

### 5. 영상 생성 모듈 (Mock)
- 실제 영상 생성 API는 아직 연동되지 않았으며, `video_generator.py`에서는 가상의 영상 파일 생성으로 흐름을 시뮬레이션

---

## 폴더 구조

```
prompt_video_agent/
├── app.py                        # Streamlit 기반 실행 파일
├── base_prompt_engine.py         # 프롬프트 생성 및 구성 요소 추출 핵심 로직
├── history_manager.py            # 수정 이력 저장/불러오기 기능
├── main.py                       # 로컬 테스트 실행용 엔트리 파일
├── prompt_diff.py                # 원본/수정된 프롬프트 비교 시각화
├── prompt_history_recommender.py # 과거 입력과 유사한 문장 추천
├── prompt_selector.py            # 생성 vs 추천 프롬프트 선택
├── video_generator.py            # 영상 생성 or mock 영상 출력
├── requirements.txt              # 의존성 목록
└── README.md                     # 설명 문서
```

---

## 실행 방법 (PyCharm 기준)

1. **가상환경 생성**
   PyCharm의 Settings → Project: prompt_video_agent → Python Interpreter → Add Interpreter
   - `New environment using Conda` 또는 `venv` 선택
   - 환경 이름: `prompt_video_agent`

2. **필수 패키지 설치**
   `requirements.txt`를 PyCharm 내 터미널에서 실행
   ```bash
   pip install -r requirements.txt
   ```

3. **실행**
   `main.py`를 선택한 후 상단 ▶ 버튼 클릭 또는 터미널에서 실행
   ```bash
   streamlit run app.py
   ```

---

## 예시

입력:
```
귀여운 강아지가 공원에서 뛰 노는 영상
```

출력:
```
귀여운 강아지가 공원에서 뛰는 장면이 담긴 즐거운 분위기의 10초 영상
```
