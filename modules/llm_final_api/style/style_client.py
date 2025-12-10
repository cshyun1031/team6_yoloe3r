from google import genai
from google.genai import types
import mimetypes


def run_style_model(api_key, model_name, image_path, prompt):
    """
    스타일 변경용 Gemini 이미지 모델을 호출하고,
    응답에서 첫 번째 이미지 파트를 찾아 바이트로 돌려준다.
    이미지가 없으면 RuntimeError를 던진다.
    """
    client = genai.Client(api_key=api_key)

    # 1. 입력 이미지 읽기
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    # 2. 모델 호출
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type=mime_type
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=1.2,
            top_p=0.9,
        )
    )

    # 3. 응답에서 이미지 파트 찾기
    image_bytes = None

    # (1) response.parts 형태로 오는 경우
    if getattr(response, "parts", None):
        for part in response.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                image_bytes = part.inline_data.data
                break

    # (2) 후보(candidates) 안의 content.parts 형태로 오는 경우
    if image_bytes is None and getattr(response, "candidates", None):
        for cand in response.candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    break
            if image_bytes is not None:
                break

    # 4. 이미지가 끝내 없으면 에러로 처리
    if image_bytes is None:
        # 텍스트로 뭔가 설명을 했을 수도 있으니, 디버깅용으로 남겨두기
        try:
            msg = response.text
        except Exception:
            msg = str(response)
        raise RuntimeError(f"모델이 이미지를 반환하지 않았습니다. 응답 내용: {msg}")

    return image_bytes
