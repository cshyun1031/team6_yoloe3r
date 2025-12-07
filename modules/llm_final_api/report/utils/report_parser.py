import re
from typing import Dict, Any, List, Union

def parse_report_output(result_text: str) -> Dict[str, Union[str, Dict, List]]:
    llm_output = result_text
    parsed_data: Dict[str, Any] = {}

    # ------ 전체적인 분위기 한 줄 ------
    match_style = re.search(
        r"#\s*전체적인 분위기는\s*\*\*(.*?)\s*스타일\*\*",
        llm_output,
        re.DOTALL, 
    )

    if match_style: 
        general = match_style.group(1).strip()
        parsed_data["general_style"] = general

        # {분위기1}, {분위기2} ,{분위기3} 추출
        # 예: "아늑하고 내추럴한 모던 스타일"에서 단어만 추출
        # '하고', '한'으로 끝나는 단어 또는 마지막 단어 추출
        moods = re.findall(r"([가-힣\s]+?)(?:하고|한|며|\s*$)", general)
        parsed_data["mood_words"] = [m.strip() for m in moods if m.strip()]

    # ------ ## 1. 분위기 정의 및 유형별 확률 ------
    mood_section_match = re.search(
        r"##\s*1\. 분위기 정의 및 유형별 확률(.*?)(?=##\s*2\. 분위기 판단 근거)",
        llm_output,
        re.DOTALL,
    )
    
    parsed_data["mood_details"] = []
    
    if mood_section_match:
        mood_section_content = mood_section_match.group(1).strip()
        
        # 각 항목을 파싱하는 패턴: - {단어}({숫자}%): {설명}
        bullet_pattern = re.compile(
            r"^\s*-\s*(?P<word>[^()]+)\((?P<percentage>\d+)%\):\s*(?P<description>.+)$",
            re.MULTILINE,
        )
        
        for bullet in bullet_pattern.finditer(mood_section_content):
            parsed_data["mood_details"].append(
                {
                    "word": bullet.group("word").strip(),
                    "percentage": int(bullet.group("percentage")),
                    "description": bullet.group("description").strip(),
                }
            )

    # ------ ## 2. 분위기 판단 근거 (원형 유지) ------
    # 이 섹션은 파싱하지 않고 텍스트 그대로 유지

    # =========================================================================
    # ------ ## 3. 가구 추천 (파싱 로직 수정: ### 헤더와 **강조** 반영) ------
    # =========================================================================
    rec_section_match = re.search(
        r"##\s*3\.\s*가구 추천(.*?)(\n##\s*4\.\s*이런 스타일 어떠세요\?|\Z)",
        llm_output,
        re.DOTALL,
    )
    
    parsed_data["recommendations_add"] = []
    parsed_data["recommendations_remove"] = []
    parsed_data["recommendations_change"] = []

    if rec_section_match:
        body = rec_section_match.group(1).strip()
        
        # 1. 추가 가구 추천: ### 현재 분위기에 맞춰 추가하면 좋을 가구 추천
        # - **아이템** : 이유 패턴 반영
        add_match = re.search(
            r"###\s*현재 분위기에 맞춰 추가하면 좋을 가구 추천\s*\n\s*-\s*\*\*(?P<item>[^*]+?)\*\*\s*:\s*(?P<reason>.+?)(\n\n|### 제거하면 좋을 가구 추천|\Z)",
            body,
            re.DOTALL | re.MULTILINE
        )
        if add_match:
            parsed_data["recommendations_add"].append({
                "item": add_match.group("item").strip(),
                "reason": add_match.group("reason").strip()
            })

        # 2. 제거 가구 추천: ### 제거하면 좋을 가구 추천
        # - **아이템** : 이유 패턴 반영
        rem_match = re.search(
            r"###\s*제거하면 좋을 가구 추천\s*\n\s*-\s*\*\*(?P<item>[^*]+?)\*\*\s*:\s*(?P<reason>.+?)(\n\n|### 분위기별 바꿨으면 하는 가구 추천|\Z)",
            body,
            re.DOTALL | re.MULTILINE
        )
        if rem_match:
            parsed_data["recommendations_remove"].append({
                "item": rem_match.group("item").strip(),
                "reason": rem_match.group("reason").strip()
            })

        # 3. 변경 가구 추천: ### 분위기별 바꿨으면 하는 가구 추천
        # - **기존** -> **추천** : 이유 패턴 반영, 두 번째 항목이 있더라도 첫 번째 항목에서 멈추도록 수정
        change_match = re.search(
            r"###\s*분위기별 바꿨으면 하는 가구 추천\s*\n\s*-\s*\*\*(?P<from_item>[^*]+?)\*\*\s*->\s*\*\*(?P<to_item>[^*]+?)\*\*\s*:\s*(?P<reason>.+?)(\n\s*-|\Z)",
            body,
            re.DOTALL | re.MULTILINE
        )
        if change_match:
            parsed_data["recommendations_change"].append({
                "from_item": change_match.group("from_item").strip(),
                "to_item": change_match.group("to_item").strip(),
                "reason": change_match.group("reason").strip()
            })
    # =========================================================================

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
            # - **{스타일}** : {이유} 형식에 맞게 수정
            bullet_pattern = re.compile(
                r"^\s*-\s*\*\*(?P<style>[^*]+?)\*\*\s*:\s*(?P<reason>.+)$",
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
    sum_section_match = re.search(
        r"##\s*정리(.*)", 
        llm_output, 
        re.DOTALL
    )
    
    if sum_section_match:
        summary_content = sum_section_match.group(1).strip()
        parsed_data["summary_section"] = summary_content
    
    return parsed_data
