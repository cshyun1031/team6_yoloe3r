import time
import json
# 1. 상대 경로 임포트를 절대 경로 임포트로 수정
from config import * from report.utils.report_parser import parse_report_output
from report.report_client import run_report_model
from report.report_prompt import report_prompt
from ultralytics import YOLOE # select_best_image 로직을 YOLOE로 대체했으므로 
import shutil
from typing import Dict, Any

# =========================================================================
# 수정된 함수: 요약 리포트 파일 생성 (토글 구조 반영)
# =========================================================================
def create_summary_report_file(parsed_data: Dict[str, Any], raw_report_text: str):
    """
    파싱된 데이터를 기반으로 Gradio UI에 보여줄 요약 리포트 템플릿을 생성합니다.
    필수 항목만 간략하게 작성하며, 상세 내용은 <details><summary> 태그 안에 전체 원본 텍스트를 포함합니다.
    """
    
    # 데이터 구조 확인 및 기본값 설정
    mood_details = parsed_data.get("mood_details", [])
    mood1 = mood_details[0] if len(mood_details) > 0 else {"word": "{분위기1}", "percentage": "{확률1}"}
    mood2 = mood_details[1] if len(mood_details) > 1 else {"word": "{분위기2}", "percentage": "{확률2}"}
    mood3 = mood_details[2] if len(mood_details) > 2 else {"word": "{분위기3}", "percentage": "{확률3}"}

    rec_add = parsed_data.get("recommendations_add", [])
    add_item = rec_add[0].get("item", "{추가 가구}") if rec_add else "{추가 가구}"

    rec_rem = parsed_data.get("recommendations_remove", [])
    rem_item = rec_rem[0].get("item", "{제거 가구}") if rec_rem else "{제거 가구}"

    rec_change = parsed_data.get("recommendations_change", [])
    if rec_change:
        change_item = rec_change[0].get("from_item", "{변경 가구}")
        rec_item = rec_change[0].get("to_item", "{추천 가구}")
    else:
        change_item = "{변경 가구}"
        rec_item = "{추천 가구}"

    rec_styles = parsed_data.get("recommended_styles", [])
    rec_style = rec_styles[0].get("style", "{추천 분위기}") if rec_styles else "{추천 분위기}"

    summary_content = f"""
# 전체적인 분위기는 **{parsed_data.get("general_style", "{분위기1}하고 {분위기2}한 {분위기3}")} 스타일**입니다.

## 1. 분위기 정의 및 유형별 확률
* **{{"{mood1['word']}"}}({mood1['percentage']}%)**: 
* **{{"{mood2['word']}"}}({mood2['percentage']}%)**: 
* **{{"{mood3['word']}"}}({mood3['percentage']}%)**: 

## 2. 가구 추가 / 제거 / 변경 추천
3-1 **현재 분위기에 맞춰 추가하면 좋을 가구 추천**
* **{add_item}** : 

3-2 **제거하면 좋을 가구 추천**
* **{rem_item}** : 

3-3 **분위기별 바꿨으면 하는 가구 추천**
* **{change_item} -> {rec_item}** : 

## 3. 이런 스타일 어떠세요? 
**{rec_style}** : 

<details>
<summary>**상세 분석 및 추천 근거 (전체 리포트 보기)**</summary>

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
        report_output_path = "report_analysis_result.txt"
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(raw_report_text)

        # 2. 파싱된 JSON 파일 저장
        parsed_json_path = "parsed_report.json"
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
        # 3. 요약 리포트 파일 생성 (report_summarize.txt)
        create_summary_report_file(parsed_data, raw_report_text)


    except Exception as e:
        print(f"2단계 (리포트 분석) 중 에러 발생: {e}")
        return


if __name__ == "__main__":
    # 2. main_report 함수 호출 시 인수를 전달
    try:
        # config.py에 정의된 변수를 사용한다고 가정
        main_report(INITIAL_IMAGE_PATHS)
    except NameError:
        print("오류: 'INITIAL_IMAGE_PATHS' 변수를 config.py에서 찾을 수 없습니다. config.py 파일과 변수 이름을 확인하세요.")
    except Exception as e:
        print(f"스크립트 실행 중 예상치 못한 에러 발생: {e}")
