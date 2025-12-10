import os
from google import genai
from google.genai import types
from PIL import Image # 이미지 저장 및 처리 
import io # 바이트 스트림 처리 
from .config import API_KEY, STYLE_MODEL

LAST_VIEW_PROMPTS = {}

def make_one_image_to_three(api_key: str, model_name: str, input_image_path: str):
    """
    앞선 과정에서 생성된 방 이미지를 입력받아,
    왼쪽 측면 뷰(Left View)와 오른쪽 측면 뷰(Right View) 2장의 이미지를 추가로 생성합니다.

    Args:
        api_key (str): Google GenAI API Key
        model_name (str): 사용할 모델명 (config.py의 STYLE_MODEL, 예: 'gemini-2.5-flash-image')
        input_image_path (str): 앞선 과정에서 생성된 원본 이미지 경로
    """
    # 1. 클라이언트 초기화.
    client = genai.Client(api_key=api_key)

    # 2. 원본 이미지 파일 읽기. (바이트 변환)
    # LLM에게 원본 이미지(레퍼런스)를 '입력'으로 제공하여, 동일한 구조와 스타일을 유지하라는 컨텍스트를 부여하기 위함
    # 즉, 1번 과정에서 만든 이미지를 모델에게 "이 공간을 기반으로 그려줘"라고 전달하기 위함.
    try:
        with open(input_image_path, "rb") as f:
            img_bytes = f.read()
    except FileNotFoundError:
        print(f" 오류: 입력 이미지를 찾을 수 없습니다. 경로를 확인하세요: {input_image_path}")
        return

    # 3. 생성할 이미지 설정. (방향, 파일명, 각도별 추가 프롬프트)

    # (1) 절대 규칙. (공통)
    base_rules = """
        당신은 실사 사진을 감쪽같이 편집하는 최고 수준의 전문 디지털 아티스트이자 포토그래머입니다.
        입력된 “단일 방 사진”을 바탕으로, 동일한 방을 유지한 상태에서
        수평(Yaw) 각도만 다르게 적용한 실사 이미지를 생성해야 합니다.

        ────────────────────────────────────────
        절대 규칙 (어떠한 경우에도 위반 금지):
        • 출력 결과는 반드시 "단일 이미지 1장".
        • 가구의 종류, 개수, 색상, 위치, 조명, 벽 색, 바닥, 창문, 방의 구조, 재질과 텍스쳐 등은 절대 바꾸지 마세요.
        • 새로운 가구, 장식, 문, 창문을 추가하지 마세요.
        • 카메라의 물리적 위치 고정.
        • 수평(Yaw) 회전만 적용.
        • 상하 각도(Pitch) 변화 없음 → tilt = 0
        • 줌인/줌아웃 금지 → zoom = 0 (원본 화각 그대로 유지)
        • 출력되는 이미지는 "실사 사진(realistic photo)"이어야 하며, 3D 렌더링·그림 느낌 금지.
        • 이 이미지는 반드시 원본과 같은 방이어야 하며, 다른 방처럼 보이게 만들면 안 됩니다.
        • 시점이 달라져서 가구의 일부가 가려지거나, 보이는 면/각도가 달라지는 정도는 자연스러운 변화이므로 허용합니다.
        • 하지만 카메라 회전을 핑계로 전체 구도나 공간의 구조, 가구 구성이 크게 달라지면 안 됩니다. 
    """

    # (2) 방향별 세부 지시. (Left / Right)
    # 기존 실험에서는 -30°, 정면, 30°가 가장 잘 나왔음
    # 모델이 gemini-3.0-flash로 변경되면서 최적 각도가 달라짐
    # 현재는 30°와 가장 유사한 40°를 사용하도록 프롬프트를 설정.
    prompts_by_direction = {
        "left": """
        [작업: 왼쪽(Yaw -40°)]
        • 카메라가 제자리에서 수평으로 왼쪽으로 약 -40° 회전한 시야.
        • 사용자가 몸을 왼쪽으로 40° 돌린 시점과 동일.

        prompt:
        camera rotated 40 degrees to the left, same room, same furniture layout,
        single frame, single viewpoint, realistic indoor photograph
        """,

        "right": """
        [작업: 오른쪽(Yaw +40°)]
        • 카메라가 제자리에서 수평으로 오른쪽으로 약 +40° 회전한 시점.
        • 사용자가 몸만 오른쪽으로 40° 돌려 같은 방을 바라본 시야.

        prompt:
        camera rotated 40 degrees to the right, same room, same furniture layout,
        single frame, single viewpoint, realistic indoor photograph
        """
    }

    # (3) 네거티브 프롬프트. (공통)
    # 생성 품질을 저해하거나 일관성을 깨는 요소를 적극적으로 배제하기 위한 지침
    negative_prompt = """
    negative prompt:
    collage, split screen, multi-view, panorama, fisheye, lens distortion,
    illustration, drawing, CGI, painting, text, watermark, subtitles,
    distorted furniture, changing room structure,
    mirrored, mirror image, left-right flip, horizontal flip, vertical flip,
    perfectly symmetric room, perfectly mirrored room,
    identical to original image, exact copy of original, no change from original,
    좌우대칭, 좌우 반전, 원본과 완전히 동일한 사진
    zoomed in, zoom-in, zoom in,
    zoomed out, zoom-out, zoom out,
    extreme close-up, close-up, tight framing,
    crop, cropped, heavily cropped, partial frame
    """

    # 각도 저장 파일명 : img4new3r_left.png, img4new3r_right.png.
    tasks = [
        {"direction": "left", "filename": "img4new3r_left.png"},
        {"direction": "right", "filename": "img4new3r_right.png"}
    ]

    # 4. 왼쪽/오른쪽 이미지 생성 반복.
    for task in tasks:
        direction = task["direction"]
        output_filename = task["filename"]
        
        # 현재 방향에 맞는 프롬프트 조립.
        # [공통규칙] + [현재 방향 지시] + [네거티브]를 결합하여 모델에 전달
        final_prompt = base_rules + "\n" + prompts_by_direction[direction] + "\n" + negative_prompt
        LAST_VIEW_PROMPTS[direction] = final_prompt

        print(f"⏳ '{direction}' 측면 이미지 생성 중...")


        try:
            # 5. 모델 호출. (이미지 생성 요청)
            # contents 인자에 '레퍼런스 이미지'와 '텍스트 프롬프트'를 모두 전달하여 
            # Gemini의 이미지 참조 및 생성 능력을 활용.
            # Vertex AI Studio 설정: 온도 0.1 이하, 이미지 출력.
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    # 1번 이미지.(레퍼런스)
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg"  # 또는 image/png, 입력 파일에 맞춰 조정 가능.
                    ),
                    # 텍스트 프롬프트.
                    final_prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,  # 온도 설정 (0.3): 온도가 낮을수록 결과물의 일관성이 높아지고 창의성을 낮아짐.
                    top_p=0.3         # Top-P 누클리어스 샘플링 설정 (0.3) : Top-P 값이 낮을수록 확률이 높은 답을 내놓을 가능성을 높아짐.
                    # 모델이 이미지를 반환하도록 설정. (모델 스펙에 따라 파라미터가 다를 수 있음)
                    # 만약 순수 Imagen 모델이라면 generate_images 메서드를 써야 할 수도 있음.
                    # 여기서는 Gemini 멀티모달(입력:이미지+텍스트 -> 출력:이미지)을 가정.
                )
            )
            
            # 6. 응답 처리 및 이미지 저장.
            # Gemini 모델은 이미지 생성 결과를 response.parts 내의 inline_data로 반환
            if response.parts:
                for part in response.parts:
                    # 바이너리 데이터(이미지)가 있는지 확인.
                    if part.inline_data:
                        image_data = part.inline_data.data
                        
                        # 바이트 데이터를 이미지 파일로 저장.
                        img = Image.open(io.BytesIO(image_data))
                        filepath = os.path.join('apioutput',output_filename)
                        img.save(filepath)
                        print(f"   저장 완료: {filepath}")
                        break # 이미지를 하나 찾으면 저장하고 다음 태스크로.
                else:
                    # 루프가 break 없이 끝났다면 이미지가 없다는 뜻.
                    print(f"    경고: 모델 응답에 이미지 데이터가 없습니다. (텍스트 응답일 수 있음)")
                    print(f"   응답 내용: {response.text}")
            else:
                print(f"   오류: 모델로부터 응답이 비어있습니다.")

        except Exception as e:
            print(f"   '{direction}' 이미지 생성 중 에러 발생: {e}")

    print("\n 모든 추가 뷰 이미지 생성 작업이 끝났습니다.")

# 테스트용 실행 코드 (이 파일을 직접 실행할 때만 동작.)
if __name__ == "__main__":
    # NOTE: 이 블록이 동작하려면, 이 파일이 config.py와 같은 레벨에 있다고 가정하며, 
    #       config.py에서 API_KEY와 STYLE_MODEL을 import해야 합니다. 
    #       프로젝트 최종 구조에 따라 main 함수에서 호출하는 것이 일반적이므로,
    #       이 테스트 코드는 제거하거나 주석 처리하는 것이 좋습니다.
    from config import API_KEY, STYLE_MODEL # 테스트를 위해 임시 import

    TEST_INPUT_PATH = "styled_output.jpg" # 1번 과정 결과물이 있다고 가정.
    
    # 파일이 존재할 때만 테스트.
    if os.path.exists(TEST_INPUT_PATH):
        make_one_image_to_three(API_KEY, STYLE_MODEL, TEST_INPUT_PATH)
    else:
        print(f"테스트를 위한 입력 파일({TEST_INPUT_PATH})이 없습니다.")

        print("먼저 1번 과정(스타일 변환)을 실행하여 이미지를 생성해주세요.")

