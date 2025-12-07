# modules/llm_final_api/main_report.py

import time
import json
# 1. 상대 경로 임포트를 절대 경로 임포트로 수정
from .config import * 
from .report.utils.report_parser import parse_report_output
from .report.report_client import run_report_model
from .report.report_prompt import report_prompt
from ultralytics import YOLOE # select_best_image 로직을 YOLOE로 대체했으므로 
import shutil
from typing import Dict, Any
import re # 정규표현식 임포트 추가

# =========================================================================
# 수정된 함수: 요약 리포트 파일 생성 (추천 분위기 파싱 실패 시 대체 로직 추가 및 별표 포맷 수정)
# =========================================================================
def create_summary_report_file(parsed_data: Dict[str, Any], raw_report_text: str):
    """
    파싱된 데이터를 기반으로 Gradio UI에 보여줄 요약 리포트 템플릿을 생성합니다.
    요청하신 report_summarize.txt 형식에 맞춰 내용을 구성하며,
    LLM 출력의 잔여 플레이스홀더({})를 제거하는 로직을 추가하고,
    '추천 분위기' 파싱 실패 시 raw_report_text에서 직접 추출하는 대체 로직을 추가합니다.
    """
    
    def clean_brace(text: Any, default_text: str) -> str:
        """파싱된 텍스트에서 중괄호를 제거하고 기본값을 반환합니다."""
        if text is None:
            return default_text
        # LLM의 잘못된 출력 (예: "{보헤미안}")을 처리하기 위해 중괄호 제거
        return str(text).strip().replace('{', '').replace('}', '')

    # 데이터 구조 확인 및 기본값 설정
    mood_details = parsed_data.get("mood_details", [])
    
    # 1. 분위기 정의 및 유형별 확률 데이터 추출 및 중괄호 제거
    mood1_word = clean_brace(mood_details[0].get("word") if len(mood_details) > 0 else None, "{분위기1}")
    mood1_percent = str(mood_details[0].get("percentage", "{확률1}")) if len(mood_details) > 0 else "{확률1}"
    
    mood2_word = clean_brace(mood_details[1].get("word") if len(mood_details) > 1 else None, "{분위기2}")
    mood2_percent = str(mood_details[1].get("percentage", "{확률2}")) if len(mood_details) > 1 else "{확률2}"
    
    mood3_word = clean_brace(mood_details[2].get("word") if len(mood_details) > 2 else None, "{분위기3}")
    mood3_percent = str(mood_details[2].get("percentage", "{확률3}")) if len(mood_details) > 2 else "{확률3}"

    # 2. 가구 추천 데이터 추출 및 중괄호 제거
    rec_add = parsed_data.get("recommendations_add", [])
    # report_parser.py에서 첫 번째 항목만 파싱하므로, 여기서도 첫 번째 항목만 사용
    add_item = clean_brace(rec_add[0].get("item") if rec_add and rec_add[0].get("item") else None, "{추가 가구}")

    rec_rem = parsed_data.get("recommendations_remove", [])
    rem_item = clean_brace(rec_rem[0].get("item") if rec_rem and rec_rem[0].get("item") else None, "{제거 가구}")

    rec_change = parsed_data.get("recommendations_change", [])
    # 요청에 따라 첫 번째 변경 추천 항목만 사용 (rec_change[0])
    if rec_change and rec_change[0].get("from_item") and rec_change[0].get("to_item"):
        change_item = clean_brace(rec_change[0].get("from_item"), "{변경 가구}")
        rec_item = clean_brace(rec_change[0].get("to_item"), "{추천 가구}")
    else:
        change_item = "{변경 가구}"
        rec_item = "{추천 가구}"

    # 3. 추천 스타일 데이터 추출 및 중괄호 제거
    rec_styles = parsed_data.get("recommended_styles", [])
    rec_style_default = "{추천 분위기}"
    # report_parser.py에서 첫 번째 항목만 파싱하므로, 여기서도 첫 번째 항목만 사용
    rec_style_val = rec_styles[0].get("style") if rec_styles and rec_styles[0].get("style") else None
    rec_style = clean_brace(rec_style_val, rec_style_default)
    
    # ===== '추천 분위기' 파싱 실패 시 대체 로직 추가 (report_parser.py 수정으로 필요 없을 수 있지만 안전을 위해 유지) =====
    if rec_style == rec_style_default:
        # ## 4. 이런 스타일 어떠세요? 섹션의 첫 번째 항목 추출 시도
        match = re.search(r"##\s*4\.\s*이런 스타일 어떠세요\?\s*\n-\s*\*\*(.*?)\*\*\s*:\s*", raw_report_text, re.DOTALL)
        if match:
            # 스타일명만 추출 (예: '스칸디나비아 + 보헤미안 퓨전' 추출)
            rec_style = clean_brace(match.group(1).strip(), rec_style_default)
    # ===================================================================

    # 전체 분위기 스타일 (general_style) 추출 및 중괄호 제거
    general_style_raw = parsed_data.get("general_style", "{분위기1}하고 {분위기2}한 {분위기3}")
    general_style = clean_brace(general_style_raw, "{분위기1}하고 {분위기2}한 {분위기3}")

    # =====================================================================
    # 최종 요약 콘텐츠 생성 (요청된 형식 및 별표 포맷 수정 적용)
    # =====================================================================
    summary_content = f"""

1. 전체적 분위기 : {mood1_word} ({mood1_percent}%), {mood2_word} ({mood2_percent}%), {mood3_word} ({mood3_percent}%)

2. 가구 추가 / 제거 / 변경 추천
### 가구 추가: **{add_item}** 
### 가구 제거: **{rem_item}** 
### 가구 변경: **{change_item} -> {rec_item}**

3. 이런 스타일 어떠세요?
### **{rec_style}**



<details>

<summary> 상세 분석 및 추천 근거 (전체 리포트 보기)</summary>

{raw_report_text}

</details>
"""
    summary_output_path = "report_summarize.txt"
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(summary_content.strip())
    
    print(f"요약 리포트 파일 생성 완료: {summary_output_path}")
    return summary_output_path
# =========================================================================


def main_report(img_path):
    # ----- 1단계: YOLOE를 이용한 최적의 입력 이미지 1장 선택 ------
    model = YOLOE("yoloe-11s-seg.pt")
    max_cnt = 0
    max_idx = 0
    
    # YOLOE 예측은 시간이 걸릴 수 있으므로, 단일 이미지 리스트를 기대합니다.
    for i, img in enumerate(img_path):
        # YOLOE 모델을 사용하여 바운딩 박스 개수 확인
        results = model.predict(img)
        # results[0].boxes는 DetBoxes 객체이며, len()으로 바운딩 박스 개수를 얻습니다.
        current_cnt = len(results[0].boxes) 
        
        if current_cnt > max_cnt:
            max_idx = i
            max_cnt = current_cnt
        # else: pass (생략 가능)

    final_input_path = img_path[max_idx]
    print('최적 입력 이미지 : ' + final_input_path)
    shutil.copyfile(img_path[max_idx], SELECTED_IMAGE_PATH)

    # ------ 2단계: 공간 분석 리포트 생성 ------
    try:
        # Gemini에 이미지 + 분석용 프롬프트 전달
        raw_report_text = run_report_model(
            api_key=API_KEY,
            model_name=REPORT_MODEL,
            image_path=final_input_path,
            prompt=report_prompt,
        )

        time.sleep(1)

        # 텍스트 분석 및 파싱
        parsed_data = parse_report_output(raw_report_text)

        # 1. 원본 리포트 파일 저장 (기존 report_analysis_result.txt)
        # 요청에 따라 원본 리포트 파일을 생성하는 코드는 유지하지 않으며,
        # 대신 요약 파일의 내용을 복사하여 UI의 하드코딩된 경로를 만족시킵니다.

        # 2. 파싱된 JSON 파일 저장
        parsed_json_path = "parsed_report.json"
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
        # 3. 요약 리포트 파일 생성 (report_summarize.txt)
        summary_path = create_summary_report_file(parsed_data, raw_report_text)
        
        # UI/다음 단계에서 report_analysis_result.txt를 읽는 문제를 해결하기 위한 수정
        # report_summarize.txt (요약 내용)을 UI가 기대하는 파일 경로에 복사하여 내용을 치환합니다.
        original_report_path = "report_analysis_result.txt"
        shutil.copyfile(summary_path, original_report_path) 
        
        # 이 부분이 UI/다음 단계에 전달될 최종 아웃풋 파일 경로입니다.
        return summary_path 

    except Exception as e:
        print(f"2단계 (리포트 분석) 중 에러 발생: {e}")
        return None # 오류 발생 시 None 반환


if __name__ == "__main__":
    # 2. main_report 함수 호출 시 인수를 전달
    try:
        # config.py에 정의된 변수를 사용한다고 가정
        main_report(INITIAL_IMAGE_PATHS)
    except NameError:
        print("오류: 'INITIAL_IMAGE_PATHS' 변수를 config.py에서 찾을 수 없습니다. config.py 파일과 변수 이름을 확인하세요.")
    except Exception as e:
        print(f"스크립트 실행 중 예상치 못한 에러 발생: {e}")




