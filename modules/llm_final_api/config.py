# config.py 파일

# 터미널이나 명령 프롬프트에서 아래 라이브러리 설치
# pip install python-dotenv
import os
from dotenv import load_dotenv

# .env 파일을 만드세요
# 만드는 방법: 문서 이름을 .env 로 설정하시고,
# 거기에 API_KEY="YOUR_GEMINI_API_KEY_HERE" 이것만 실제 api 키로 바꿔 써놓으시면 됩니다.
# --- .env 파일 로드 ---
# 프로젝트 루트 폴더에 있는 .env 파일을 읽어 환경 변수를 로드합니다.
load_dotenv() 

# --- API 키 설정 ---
# 환경 변수에서 API_KEY 값을 가져와 적용합니다.
# 만약 .env 파일에 키가 없다면, 두 번째 인수의 값(None)이 사용됩니다.
API_KEY = os.getenv("API_KEY") 

# API 키가 제대로 로드되었는지 확인 (선택 사항)
if not API_KEY:
    print("🚨 경고: .env 파일에서 API_KEY를 로드하지 못했습니다. 키 값을 확인하세요.")

# 3장의 최초 입력 이미지 경로
# 이때 이미지는 각각 왼쪽 30도(-30), 정면, 오른쪽 30도(30)
INITIAL_IMAGE_PATHS = [
    "C:/Users/minso/Desktop/report_3imgs/room_30degree.jpg", 
    "C:/Users/minso/Desktop/report_3imgs/room_original.jpg",
    "C:/Users/minso/Desktop/report_3imgs/room_-30degree.jpg"
]

# 3장 중 AI가 선택한 '최적 이미지'가 임시로 저장될 경로
# 이후 모든 프로세스(Report, Style)는 이 경로를 사용합니다.
SELECTED_IMAGE_PATH = "selected_input_image.jpg"




REPORT_MODEL = "gemini-2.5-flash" # 리포트 생성 모델
STYLE_MODEL = "gemini-2.5-flash-image"  # 이미지 출력 모델






