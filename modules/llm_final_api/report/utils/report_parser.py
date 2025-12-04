import re
from typing import Dict, Any, List, Union

def parse_report_output(result_text: str) -> Dict[str, Union[str, Dict, List]]:
    llm_output = result_text
    parsed_data: Dict[str, Any] = {}

    # ------ 전체적인 분위기 한 줄 (원형 유지) ------
    match_style = re.search(
        r"#\s*전체적인 분위기는\s*\*\*(.*?)\s*스타일\*\*",
        llm_output,
        re.DOTALL, 
    )

    if match_style: 
        general = match_style.group(1).strip()
        parsed_data["general_style"] = general

        # {분위기1}, {분위기2} ,{분위기3} 추출 (원형 유지)
        # "아늑하고 내추럴한 모던 스타일"에서 단어만 추출
        moods = re.findall(r"([가-힣\s]+?)(?:하고|한|\s*$)", general)
        parsed_data["mood_words"] = [m.strip() for m in moods if m.strip()]

    # ------ ## 1. 분위기 정의 및 유형별 확률 ------
    # 섹션 탐지 패턴 수정: 다음 헤딩인 '## 2.'까지
    mood_section_match = re.search(
        r"##\s*1\. 분위기 정의 및 유형별 확률(.*?)(?=##\s*2\. 분위기 판단 근거)",
        llm_output,
        re.DOTALL,
    )

    if mood_section_match:
        mood_section = mood_section_match.group(1)

        # {분위기}({확률}%):\n{설명} 패턴 정의 (줄바꿈 인식 포함)
        # - {"{분위기}"}({확률}%):\n{설명} 형식
        PATTERN_MOOD_DETAIL = re.compile(
            r'-\s*"([^"]+)"\s*\((\d+)%\):\s*\n\s*(.*?)',
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

    
    # ------ ## 2. 분위기 판단 근거 (원형 유지) -------
    basis_section_match = re.search(
        r"##\s*2\. 분위기 판단 근거(.*?)(?=##\s*3\. 가구 추가 / 제거 / 변경 추천)",
        llm_output,
        re.DOTALL,
    )
    if basis_section_match:
        basis_section = basis_section_match.group(1)

        # PATTERN_BASIS는 원본 템플릿의 `- {키} : {값}` 형식에 맞춰 유지
        PATTERN_BASIS = r"-\s*(.*?):\s*(.*)"
        basis_matches = re.findall(PATTERN_BASIS, basis_section)

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

    # --- ## 3. 가구 추가 / 제거 / 변경 추천 (통합 섹션) ---
    # 3-1, 3-2, 3-3을 묶는 상위 헤딩 ## 3. 섹션 탐지
    rec_section_match = re.search(
        r"##\s*3\. 가구 추가 / 제거 / 변경 추천(.*?)(?=##\s*4\. 이런 스타일 어떠세요\?)",
        llm_output,
        re.DOTALL,
    )

    if rec_section_match:
        rec_section = rec_section_match.group(1).strip()
        
        # 3-1: 추가 추천 패턴: - **현재 분위기에 맞춰 추가하면 좋을 가구 추천**\n**가구** :\n근거
        add_match = re.search(
            r"-\s*\*\*현재 분위기에 맞춰 추가하면 좋을 가구 추천\*\*\s*\n\s*\*\*(.*?)\*\*\s*:\s*(.*)",
            rec_section,
            re.DOTALL
        )
        if add_match:
            item, reason = add_match.groups()
            parsed_data["recommendations_add"] = [
                {"item": item.strip(), "reason": reason.strip()}
            ]
        
        # 3-2: 제거 추천 패턴: - **제거하면 좋을 가구 추천**\n**가구** :\n근거
        rem_match = re.search(
            r"-\s*\*\*제거하면 좋을 가구 추천\*\*\s*\n\s*\*\*(.*?)\*\*\s*:\s*(.*)",
            rec_section,
            re.DOTALL
        )
        if rem_match:
            item, reason = rem_match.groups()
            parsed_data["recommendations_remove"] = [
                {"item": item.strip(), "reason": reason.strip()}
            ]

        # 3-3: 변경 추천 패턴: - **분위기별 바꿨으면 하는 가구 추천**\n**변경 -> 추천** :\n근거
        change_match = re.search(
            r"-\s*\*\*분위기별 바꿨으면 하는 가구 추천\*\*\s*\n\s*\*\*(.*?)\s*->\s*(.*?)\*\*\s*:\s*(.*)",
            rec_section,
            re.DOTALL
        )
        if change_match:
            src, dst, reason = change_match.groups()
            parsed_data["recommendations_change"] = [
                {
                    "from_item": src.strip(), 
                    "to_item": dst.strip(), 
                    "reason": reason.strip()
                }
            ]

    # ------ ## 4. 이런 스타일 어떠세요? ------
    # 섹션 탐지 패턴 수정: 다음 헤딩인 '## 정리'까지
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
            # - **{스타일}** :\n {이유} 형식
            bullet_pattern = re.compile(
                r"^\s*\*\*(?P<style>[^:]+?)\*\*\s*:\s*(?P<reason>.+)$",
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

    # ------ ## 정리 (원형 유지) ------
    sum_section_match = re.search(r"##\s*정리(.*)", llm_output, re.DOTALL)
    if sum_section_match:
        sum_section = sum_section_match.group(1)
        lines = re.findall(r"-\s*(.*)", sum_section)

        parsed_data["summary"] = {}
        for idx, sentence in enumerate(lines):
            key = f"summary{idx + 1}"
            parsed_data["summary"][key] = sentence.strip()

    return parsed_data
