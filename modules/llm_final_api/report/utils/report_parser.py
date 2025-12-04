import re
from typing import Dict, Any, List, Union

def parse_report_output(result_text: str) -> Dict[str, Union[str, Dict, List]]:
    llm_output = result_text
    parsed_data: Dict[str, Any] = {}

    # 참고: 모든 re.search에서 '##\s*' 패턴은 유지하지만, 
    # 파싱 대상 섹션 내의 패턴은 줄바꿈과 콜론 형태에 맞게 수정합니다.

    # --- 전체적인 분위기 한 줄 ---
    # '전체적인 분위기는 "{...} 스타일"입니다.' 패턴에 맞춤 (## 제외된 첫 줄)
    match_style = re.search(
        r"전체적인 분위기는\s*\"(.*?)\"\s*스타일",
        llm_output,
        re.DOTALL, 
    )

    if match_style: 
        general = match_style.group(1).strip()
        parsed_data["general_style"] = general
        
        # "{분위기1}하고" "{분위기2}한" "{분위기3}" 패턴에서 단어 추출
        moods = [m.strip().replace('하고', '').replace('한', '').replace('"', '').strip() 
                 for m in re.findall(r'\"([^\"]+?)\"', general)]
        
        parsed_data["mood_words"] = [m for m in moods if m]

    # --- ## 1. 분위기 정의 및 유형별 확률 ---
    # 헤딩에 '##\s*1.' 패턴 유지. 다음 헤딩인 '##\s*2.'까지 추출
    mood_section_match = re.search(
        r"##\s*1\. 분위기 정의 및 유형별 확률(.*?)(?=##\s*2\. 분위기 판단 근거)",
        llm_output,
        re.DOTALL,
    )

    if mood_section_match:
        mood_section = mood_section_match.group(1).strip()

        # {분위기}({확률}%):\n{설명} 패턴에 맞춤
        PATTERN_MOOD_DETAIL = re.compile(
            r"([^\(]+?)\s*\((\d+)%\):\s*\n\s*(.+?)",  # 분위기 이름 (확률)%:\n 설명
            re.DOTALL
        )

        mood_matches = PATTERN_MOOD_DETAIL.findall(mood_section)
        parsed_data["mood_details"] = []

        for mood, pct, desc in mood_matches:
            parsed_data["mood_details"].append(
                {
                    "word": mood.strip(),
                    "percentage": int(pct),
                    "description": desc.strip(),
                }
            )

    
    # --- ## 2. 분위기 판단 근거 ---
    # 헤딩에 '##\s*2.' 패턴 유지. 다음 헤딩인 '##\s*3-1.'까지 추출
    basis_section_match = re.search(
        r"##\s*2\. 분위기 판단 근거(.*?)(?=##\s*3-1\. 현재 분위기에 맞춰 추가하면 좋을 가구 추천)",
        llm_output,
        re.DOTALL,
    )
    if basis_section_match:
        basis_section = basis_section_match.group(1).strip()

        # - {키} :\n {값} 패턴에 맞춤
        PATTERN_BASIS = re.compile(
            r"-\s*([가-힣\s]+?)\s*:\s*\n\s*(.*?)",
            re.DOTALL
        )
        basis_matches = PATTERN_BASIS.findall(basis_section)

        parsed_data["basis"] = {}
        key_mapping = {
            "가구 배치 및 공간 분석": "furniture_layout",
            "색감 및 질감": "color_texture",
            "소재": "material",
        }

        for key, value in basis_matches:
            k = key.strip()
            v = value.strip()
            if k in key_mapping:
                parsed_data["basis"][key_mapping[k]] = v
            else:
                parsed_data["basis"][k] = v

    # --- ## 3-1. 현재 분위기에 맞춰 추가하면 좋을 가구 추천 ---
    basis_section_match = re.search(
        r"##\s*3-1\. 현재 분위기에 맞춰 추가하면 좋을 가구 추천(.*?)(?=##\s*3-2\. 제거하면 좋을 가구 추천)",
        llm_output,
        re.DOTALL,
    )
    if basis_section_match:
        add_section = basis_section_match.group(1).strip()

        # - {추가 가구} :\n {근거} 패턴에 맞춤
        PATTERN_ADD = re.compile(
            r"-\s*([가-힣\s]+?)\s*:\s*\n\s*(.*?)",
            re.DOTALL
        )
        add_matches = PATTERN_ADD.findall(add_section)

        parsed_data["recommendations_add"] = []

        for item, reason in add_matches:
            parsed_data["recommendations_add"].append(
                {
                    "item": item.strip(),
                    "reason": reason.strip(),
                }
            )

    # --- ## 3-2. 제거하면 좋을 가구 추천 ---
    rem_section_match = re.search(
        r"##\s*3-2\. 제거하면 좋을 가구 추천(.*?)(?=##\s*3-3\. 분위기별 바꿨으면 하는 가구 추천)",
        llm_output,
        re.DOTALL,
    )
    if rem_section_match:
        rem_section = rem_section_match.group(1).strip()

        # - {제거 가구} :\n {근거} 패턴에 맞춤
        PATTERN_REM = re.compile(
            r"-\s*([가-힣\s]+?)\s*:\s*\n\s*(.*?)",
            re.DOTALL
        )
        rem_matches = PATTERN_REM.findall(rem_section)

        parsed_data["recommendations_remove"] = []
        for item, reason in rem_matches:
            parsed_data["recommendations_remove"].append(
                {
                    "item": item.strip(),
                    "reason": reason.strip(),
                }
            )

    # --- ## 3-3. 분위기별 바꿨으면 하는 가구 추천 ---
    # 다음 헤딩인 '##\s*4.'까지 추출
    change_section_match = re.search(
        r"##\s*3-3\. 분위기별 바꿨으면 하는 가구 추천(.*?)(?=##\s*4\. 이런 스타일 어떠세요\?)",
        llm_output,
        re.DOTALL,
    )
    if change_section_match:
        change_section = change_section_match.group(1).strip()

        # - {변경 가구} -> {추천 가구} :\n {근거} 패턴에 맞춤
        PATTERN_CHANGE = re.compile(
            r"-\s*([가-힣\s]+?)\s*->\s*([가-힣\s]+?)\s*:\s*\n\s*(.*?)",
            re.DOTALL
        )
        change_matches = PATTERN_CHANGE.findall(change_section)

        parsed_data["recommendations_change"] = []
        for src, dst, reason in change_matches:
            parsed_data["recommendations_change"].append(
                {
                    "from_item": src.strip(),
                    "to_item": dst.strip(),
                    "reason": reason.strip(),
                }
            )

    # --- ## 4. 이런 스타일 어떠세요? ---
    # 다음 헤딩인 '##\s*정리'까지 추출
    section_pattern = re.compile(
        r"^##\s*4\.\s*이런 스타일 어떠세요\?\s*$"
        r"(?P<body>.*?)(?=^##\s*정리|\Z)", 
        re.MULTILINE | re.DOTALL,
    )

    m = section_pattern.search(llm_output)
    
    parsed_data["recommended_styles"] = []
    
    if m:
        body = m.group("body").strip()
        
        if body:
            # - {스타일} :\n {이유} 형식
            bullet_pattern = re.compile(
                r"^\s*-\s*(?P<style>[^:]+?)\s*:\s*\n\s*(?P<reason>.+)$",
                re.MULTILINE | re.DOTALL,
            )

            for b in bullet_pattern.finditer(body):
                style = b.group("style").strip()
                reason = b.group("reason").strip()
                parsed_data["recommended_styles"].append(
                    {
                        "style": style,
                        "reason": reason,
                    }
                )

    # --- ## 정리 ---
    sum_section_match = re.search(r"##\s*정리(.*)", llm_output, re.DOTALL)
    if sum_section_match:
        sum_section = sum_section_match.group(1)

        # "- {문장}" 형태의 모든 줄을 뽑아온다.
        lines = re.findall(r"-\s*(.*)", sum_section)

        parsed_data["summary"] = {}
        # 요약이 5개라는 가정 하에 1부터 5까지 키를 지정
        for idx, sentence in enumerate(lines[:5]):
            key = f"summary{idx + 1}"
            parsed_data["summary"][key] = sentence.strip()

    # 최종 결과 반환
    return parsed_data
