# modules/llm_final_api/main_report.py

import time
import json
# 1. 상대 경로 임포트를 절대 경로 임포트로 수정
# 현재 파일(main_report.py)의 상위 패키지(modules.llm_final_api) 내 'config' 모듈에서 모든 항목(*)을 임포트.
# 'config'는 API 키, 모델 이름, 경로 등 설정값을 포함.
from .config import * 
from .report.utils.report_parser import parse_report_output
from .report.report_client import run_report_model
from .report.report_prompt import report_prompt
from ultralytics import YOLOE # select_best_image 로직을 YOLOE로 대체했으므로 
import shutil
from typing import Dict, Any
import re # 정규표현식 임포트 추가


# 추가된 부분
# 요약 리포트 파일 생성 : 추천 분위기 파싱 실패 시 대체 로직 추가 및 별표 포맷 수정
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
    mood_details = parsed_data.get("mood_details", [])  # 분위기 유형 및 확률 리스트
    
    # 1. 분위기 정의 및 유형별 확률 데이터 추출 및 중괄호 제거
    # 최대 3순위까지의 분위기 유형과 확률을 추출하고, 파싱 실패 시 기본 플레이스홀더 사용.
    mood1_word = clean_brace(mood_details[0].get("word") if len(mood_details) > 0 else None, "{분위기1}")
    mood1_percent = str(mood_details[0].get("percentage", "{확률1}")) if len(mood_details) > 0 else "{확률1}"
    
    mood2_word = clean_brace(mood_details[1].get("word") if len(mood_details) > 1 else None, "{분위기2}")
    mood2_percent = str(mood_details[1].get("percentage", "{확률2}")) if len(mood_details) > 1 else "{확률2}"
    
    mood3_word = clean_brace(mood_details[2].get("word") if len(mood_details) > 2 else None, "{분위기3}")
    mood3_percent = str(mood_details[2].get("percentage", "{확률3}")) if len(mood_details) > 2 else "{확률3}"

    # 2. 가구 추천 데이터 추출 및 중괄호 제거
    rec_add = parsed_data.get("recommendations_add", [])    # 추가 추천 리스트
    # report_parser.py에서 첫 번째 항목만 파싱했다고 가정하고, 여기서도 첫 번째 항목만 사용.
    add_item = clean_brace(rec_add[0].get("item") if rec_add and rec_add[0].get("item") else None, "{추가 가구}")

    rec_rem = parsed_data.get("recommendations_remove", [])   # 제거 추천 리스트
    rem_item = clean_brace(rec_rem[0].get("item") if rec_rem and rec_rem[0].get("item") else None, "{제거 가구}")

    rec_change = parsed_data.get("recommendations_change", [])   # 변경 추천 리스트
    # 첫 번째 변경 추천 항목만 사용 (rec_change[0])
    if rec_change and rec_change[0].get("from_item") and rec_change[0].get("to_item"):
        # 'from_item' (변경 대상)과 'to_item' (추천 대상)을 추출
        change_item = clean_brace(rec_change[0].get("from_item"), "{변경 가구}")
        rec_item = clean_brace(rec_change[0].get("to_item"), "{추천 가구}")
    else:
        # 파싱 실패 시 기본 플레이스홀더 설정
        change_item = "{변경 가구}"
        rec_item = "{추천 가구}"

    # 3. 추천 스타일 데이터 추출 및 중괄호 제거
    rec_styles = parsed_data.get("recommended_styles", [])    # 추천 스타일 리스트
    rec_style_default = "{추천 분위기}"   # 파싱 실패 시 기본값
    # report_parser.py에서 첫 번째 항목만 파싱하므로, 여기서도 첫 번째 항목만 사용
    rec_style_val = rec_styles[0].get("style") if rec_styles and rec_styles[0].get("style") else None
    rec_style = clean_brace(rec_style_val, rec_style_default)
    
    # ------ '추천 분위기' 파싱 실패 시 대체 로직 추가 (안전 장치) ------
    if rec_style == rec_style_default:
        # LLM의 원본 출력 텍스트(raw_report_text)에서 정규표현식을 사용하여
		# "## 4. 이런 스타일 어떠세요?" 섹션의 첫 번째 항목을 직접 추출 시도.
		# 정규표현식: '##' 다음에 '4.'와 '이런 스타일 어떠세요?'가 오고, 
		# 그 다음 줄바꿈 후 ' **(캡처 그룹)** : ' 패턴을 찾음.
        match = re.search(r"##\s*4\.\s*이런 스타일 어떠세요\?\s*\n-\s*\*\*(.*?)\*\*\s*:\s*", raw_report_text, re.DOTALL)
        if match:
            # 첫 번째 캡처 그룹(.*?): 스타일명 (예: '스칸디나비아 + 보헤미안 퓨전') 추출
            rec_style = clean_brace(match.group(1).strip(), rec_style_default)
    

    # 전체 분위기 스타일 (general_style) 추출 및 중괄호 제거
	# 리포트 전문의 첫 문장 등에 사용될 전반적인 스타일 요약 문구.
    general_style_raw = parsed_data.get("general_style", "{분위기1}하고 {분위기2}한 {분위기3}")
    general_style = clean_brace(general_style_raw, "{분위기1}하고 {분위기2}한 {분위기3}")

    # ------ 최종 요약 콘텐츠 생성 (요청된 형식 및 별표 포맷 수정 적용) ------
	# Markdown 형식의 요약 리포트 콘텐츠를 구성.
    summary_content = f"""

1. 방 전체 분위기 : {mood1_word} ({mood1_percent}%), {mood2_word} ({mood2_percent}%), {mood3_word} ({mood3_percent}%)

2. 가구 추가 / 제거 / 변경 추천
### **가구 추가:{add_item}** 
### **가구 제거:{rem_item}** 
### **가구 변경:{change_item} -> {rec_item}**

3. 이런 스타일 어떠세요?
### **{rec_style}**



<details>

<summary> 상세 분석 및 추천 근거 (전체 리포트 보기)</summary>

{raw_report_text}

</details>
"""
    summary_output_path = "report_summarize.txt" # 요약 리포트 파일 경로 정의
    # 정의된 경로에 요약 콘텐츠를 UTF-8 인코딩으로 저장
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(summary_content.strip())   # 불필요한 공백 제거 후 쓰기
    
    print(f"요약 리포트 파일 생성 완료: {summary_output_path}")
    return summary_output_path



def main_report(img_path):
    # ------ 1단계: YOLOE를 이용한 최적의 입력 이미지 1장 선택 ------
    model = YOLOE("yoloe-11s-seg.pt")  # YOLOE 모델 로드. 'yoloe-11s-seg.pt'는 사전 학습된 모델 가중치 파일.
    max_cnt = 0  # 탐지된 최대 객체 수 초기화
    max_idx = 0  # 최대 객체 수가 탐지된 이미지의 인덱스 초기화
    
    # YOLOE 예측은 시간이 걸릴 수 있으므로, 단일 이미지 리스트를 기대함.
	# 이미지 리스트를 순회하며 각 이미지에 대해 객체 탐지 수행
    for i, img in enumerate(img_path):
        # YOLOE 모델을 사용하여 바운딩 박스 개수 확인
        results = model.predict(img)
        # results[0].boxes는 DetBoxes 객체이며, len()으로 바운딩 박스 개수를 얻음.
        current_cnt = len(results[0].boxes) 

        # 현재 이미지의 객체 수가 최대 객체 수보다 많으면 갱신
        if current_cnt > max_cnt:
            max_idx = i   # 최적 이미지 인덱스 갱신
            max_cnt = current_cnt   # 최대 객체 수 갱신
        # else: pass (생략 가능)

    final_input_path = img_path[max_idx]   # 최종 선택된 이미지 파일 경로
    print('최적 입력 이미지 : ' + final_input_path)
    # 선택된 이미지를 UI/다음 단계에서 사용하는 경로(SELECTED_IMAGE_PATH)로 복사
    shutil.copyfile(img_path[max_idx], SELECTED_IMAGE_PATH)

    # ------ 2단계: 공간 분석 리포트 생성 ------
    try:
        # Gemini에 이미지 + 분석용 프롬프트 전달
        raw_report_text = run_report_model(
            api_key=API_KEY,   # config.py에서 임포트된 API 키
            model_name=REPORT_MODEL,   # config.py에서 임포트된 모델 이름
            image_path=final_input_path,   # YOLOE가 선택한 최적의 이미지 경로
            prompt=report_prompt,   # report_prompt.py에서 임포트된 분석 프롬프트
        )

        time.sleep(1)   # API 요청 후 잠시 대기 (서버 부하 경감 혹은 안정성 확보 목적)

        # LLM 출력 텍스트를 정형화된 데이터 구조(Dict)로 파싱
        parsed_data = parse_report_output(raw_report_text)

        # 1. 원본 리포트 파일 저장 (기존 report_analysis_result.txt)
		# 요청에 따라 원본 리포트 파일을 생성하는 코드는 유지하지 않으며,
		# 대신 요약 파일의 내용을 복사하여 UI의 하드코딩된 경로를 만족시킴.

		# 2. 파싱된 JSON 파일 저장
        parsed_json_path = "parsed_report.json"
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
        # 3. 요약 리포트 파일 생성 (report_summarize.txt)
        summary_path = create_summary_report_file(parsed_data, raw_report_text)
        
        # UI/다음 단계에서 report_analysis_result.txt를 읽는 문제를 해결하기 위한 수정.
        # report_summarize.txt (요약 내용)을 UI가 기대하는 파일 경로에 복사하여 내용을 치환.
        original_report_path = "report_analysis_result.txt"
        shutil.copyfile(summary_path, original_report_path) 
        
        # 이 부분이 UI/다음 단계에 전달될 최종 아웃풋 파일 경로.
        return summary_path 

    except Exception as e:
        print(f"2단계 (리포트 분석) 중 에러 발생: {e}")
        return None # 오류 발생 시 None 반환


if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 (모듈로 임포트되지 않았을 때) 실행되는 부분
	# 2. main_report 함수 호출 시 인수를 전달
    try:
        # config.py에 정의된 변수
        main_report(INITIAL_IMAGE_PATHS)
    except NameError:
        # 'INITIAL_IMAGE_PATHS' 변수가 config.py에 정의되지 않았을 경우 처리
        print("오류: 'INITIAL_IMAGE_PATHS' 변수를 config.py에서 찾을 수 없습니다. config.py 파일과 변수 이름을 확인하세요.")
    except Exception as e:
        # 그 외 예외 처리
        print(f"스크립트 실행 중 예상치 못한 에러 발생: {e}")






