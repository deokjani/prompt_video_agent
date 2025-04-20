from difflib import SequenceMatcher

def get_prompt_diff(original: str, edited: str) -> dict:
    """
    사용자가 수정한 프롬프트의 차이를 간단하게 강조하여 반환합니다.
    변경된 부분은 [텍스트] 형태로 표시되며, 삭제/추가 구분 없이 바뀐 부분만 보여줍니다.
    """

    # 앞뒤 공백 제거
    original = original.strip()
    edited = edited.strip()

    # 유사도 비교 객체 생성
    sm = SequenceMatcher(None, original, edited)
    ops = sm.get_opcodes()

    # 변경된 부분만 추출하여 표시
    diff_text = ""
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            continue
        # 바뀐 부분만 강조
        changed = edited[j1:j2].strip()
        if changed:
            diff_text += f"[{changed}]"

    # 차이 없을 경우 기본 메시지
    if not diff_text.strip():
        diff_text = "(No Differences Found)"

    return {"text": diff_text}
