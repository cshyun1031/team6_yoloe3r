API_KEY = "APIKEY"  # 여기에 실제 API 키 입력

# 3장의 최초 입력 이미지 경로
INITIAL_IMAGE_PATHS = [
    "modules/llm_final_api/input_image/test1.png", 
    "modules/llm_final_api/input_image/test2.png",
    "modules/llm_final_api/input_image/test3.png"
]

# 3장 중 AI가 선택한 '최적 이미지'가 임시로 저장될 경로
# 이후 모든 프로세스(Report, Style)는 이 경로를 사용합니다.
SELECTED_IMAGE_PATH = "selected_input_image.jpg"




REPORT_MODEL = "gemini-2.5-flash" # 리포트 생성 모델
STYLE_MODEL = "gemini-2.5-flash-image"  # 이미지 출력 모델
