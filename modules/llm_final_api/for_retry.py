import os
import time
import mimetypes
from typing import Callable, Optional

from google import genai
from google.genai import types


# ------ 한 작업에서 허용할 재요청 횟수 ------ 
MAX_GEN_RETRY = 1          # 이미지 생성 실패 시 추가 재시도 횟수
MAX_VALIDATE_RETRY = 1     # 검수 통과 실패 시 재시도 횟수


def ensure_image_generated(
    generate_fn: Callable[[], str],             # 이미지를 생성하고 이미지 파일 경로를 반환. 생성 실패 시 original_path 반환.
    original_path: Optional[str] = None,        # result == original_path 이고 재시도 횟수가 남아있다면 변경 실패로 판단하여 재시도. 모든 재시도 실패 시 original_path 반환.
    max_retry: int = MAX_GEN_RETRY,
    sleep_sec: float = 1.0,
) -> Optional[str]:
 
    last_error: Optional[Exception] = None
    last_result: Optional[str] = None

    for attempt in range(max_retry + 1):
        try:
            result = generate_fn()
            last_result = result
        except Exception as e:
            last_error = e
            result = None

        ok = False

        if isinstance(result, str) and result:
            # 파일 실제 존재 여부 확인
            if os.path.exists(result) and os.path.getsize(result) > 0:
                # original_path 그대로 돌아온 경우 → 마지막 시도 전까지는 실패로 간주
                if original_path is not None and result == original_path and attempt < max_retry:
                    ok = False
                else:
                    ok = True

        if ok:
            return result

        # 여기까지 왔으면 이번 시도는 실패
        if attempt < max_retry:
            time.sleep(sleep_sec)

    # 모든 시도 실패
    if last_error:
        print(f"오류 : 이미지 생성 중 에러 발생 (마지막 에러): {last_error}")
    else:
        print("오류 : 이미지 생성 결과가 기대한 조건을 만족하지 못했습니다.")

    return last_result                      # 재시도 성공 시 최종 확정된 이미지 경로, 완전히 실패 시 마지막 반환값


# ------ 프롬프트 준수 검수용 ------ 
_VALIDATOR_SYSTEM_PROMPT = """
당신은 인테리어 사진 편집 웹서비스의 품질 검수 담당자입니다.

역할:
- 사용자가 업로드한 '원본 방 사진'과
- AI가 편집한 '결과 사진'을 비교하여,
요구사항을 잘 반영했는지 확인하고 PASS 또는 FAIL 을 판정합니다.

판정 기준(예시):
1. 사용자의 요구사항이 눈에 띄게 반영되어 있는가?
2. 변화가 없거나, negative prompt가 실현되어 버렸거나, 요구사항과 무관한 변화만 있다면 FAIL.
3. 변화 대상이 아닌 영역들이 원본 방과 같은 공간 구조, 가구 배치, 시점 느낌을 유지하고 있는가?

출력 형식:
- 'PASS' 또는 'FAIL' 중 한 단어만 대문자로 출력하세요.
- 다른 말이나 설명은 쓰지 마세요.
"""


def _load_image_bytes(path: str) -> Optional[tuple[bytes, str]]:
    # 경로에서 이미지를 읽어 (bytes, mime_type) 튜플로 반환. 파일이 없으면 None.

    if not path:
        return None
    if not os.path.exists(path):
        print(f"오류 : 이미지 파일을 찾을 수 없습니다: {path}")
        return None

    with open(path, "rb") as f:
        data = f.read()

    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/png"

    return data, mime_type


def check_prompt_compliance(
    api_key: str,
    model_name: str,                                        # 검수를 진행할 모델명
    user_prompt: str,
    edited_image_path: str,
    original_image_path: Optional[str] = None,
    extra_prompts: Optional[list[str]] = None,
    max_retry: int = MAX_VALIDATE_RETRY,
) -> bool:
    # gemini-flash-2.5 모델을 사용하여 이미지가 프롬프트를 잘 지켜 편집되었는지를 판정.

    edited = _load_image_bytes(edited_image_path)
    if edited is None:
        return False
    edited_bytes, edited_mime = edited

    original_bytes = None
    original_mime = None
    if original_image_path:
        orig = _load_image_bytes(original_image_path)
        if orig is not None:
            original_bytes, original_mime = orig

    client = genai.Client(api_key=api_key)

    prompt_text = _VALIDATOR_SYSTEM_PROMPT + "\n\n[사용자 요구사항]\n" + user_prompt.strip()
    if extra_prompts:
        prompt_text += "\n\n[이미지 생성에 실제 사용된 프롬프트들]\n"
        for i, p in enumerate(extra_prompts, start=1):
            if not p:
                continue
            prompt_text += f"\n--- 프롬프트 {i} ---\n{str(p).strip()}\n"

    last_error: Optional[Exception | str] = None

    for attempt in range(max_retry + 1):
        try:
            contents = []

            # 원본 이미지가 있다면 먼저 추가
            if original_bytes is not None:
                contents.append(
                    types.Part.from_bytes(
                        data=original_bytes,
                        mime_type=original_mime,
                    )
                )

            # 편집 결과 이미지
            contents.append(
                types.Part.from_bytes(
                    data=edited_bytes,
                    mime_type=edited_mime,
                )
            )

            # 텍스트 프롬프트
            contents.append(prompt_text)

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=0.3
                ),
            )

            try:
                verdict = response.text.strip().upper()
            except Exception:
                verdict = str(response).strip().upper()

            if verdict.startswith("PASS"):
                return True
            if verdict.startswith("FAIL"):
                return False

            # PASS/FAIL 이 아닌 값이면 한 번 더 시도
            last_error = f"예상치 못한 검수 결과: {verdict}"
        except Exception as e:
            last_error = e

        if attempt < max_retry:
            continue

    print(f"오류 : 검수 호출 실패 또는 비정상 응답: {last_error}")
    # 안전하게 FAIL 처리
    
    return False
