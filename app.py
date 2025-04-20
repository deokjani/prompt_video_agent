import streamlit as st
from prompt_selector import select_prompt
from video_generator import generate_video
from history_manager import save_prompt_history
from prompt_diff import get_prompt_diff

# 페이지 설정
st.set_page_config(page_title="Prompt Video Agent", layout="wide")
st.title("프롬프트 기반 영상 생성 시스템")

# 사용자 입력
st.header("1. 사용자 입력")
user_input = st.text_input("영상의 분위기/내용을 설명해보세요", placeholder="예: 귀여운 강아지가 공원에서 뛰노는 밝은 영상")

# 프롬프트 생성 버튼
if st.button("프롬프트 생성"):
    with st.spinner("프롬프트 생성 중..."):
        result_dict, source = select_prompt(user_input)

        st.session_state.auto_prompt = result_dict["auto_prompt"]
        st.session_state.recommended_prompt = result_dict.get("llm_rewritten", "")
        st.session_state.generated = False
        st.session_state.components = result_dict.get("components", {})

        if source == "recommended":
            st.markdown("과거 유사 이력 기반으로 프롬프트가 추천되었습니다.")
        else:
            st.markdown("새로운 입력에 따라 프롬프트가 생성되었습니다.")

# 프롬프트 확인 및 수정
if "auto_prompt" in st.session_state:
    st.header("2. 프롬프트 확인 및 수정")
    st.markdown("자동 생성된 프롬프트:")
    st.code(st.session_state.auto_prompt)

    # 분위기 스타일 표시
    style_hint = st.session_state.components.get("style_hint")
    if style_hint:
        st.markdown(f"반영된 분위기 스타일: `{style_hint}`")

    st.markdown("최종 프롬프트 (수정 가능):")
    default_value = st.session_state.get("recommended_prompt", st.session_state.auto_prompt)
    edited_prompt = st.text_area("최종 프롬프트 입력", value=default_value, height=100, key="final_edit")

    diff_result = get_prompt_diff(st.session_state.auto_prompt, edited_prompt)
    st.markdown("프롬프트 변경 내용:")
    st.code(diff_result["text"])

    if st.button("영상 생성하기"):
        st.session_state.generated = True

        save_prompt_history({
            "input": user_input,
            "auto_korean_prompt": st.session_state.auto_prompt,
            "edited_korean_prompt": edited_prompt,
            "diff_text": diff_result["text"]
        })

        generate_video(edited_prompt)

# 결과 영상 출력
if st.session_state.get("generated"):
    st.header("3. 생성된 영상")
    st.video("videos/placeholder.mp4")
    st.success("영상이 생성되었습니다.")
