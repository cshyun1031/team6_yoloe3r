import os
import json
import shutil

from .config import (
    API_KEY,
    STYLE_MODEL,
    SELECTED_IMAGE_PATH,
    REPORT_MODEL,
)

from .edit.image_edit import run_image_edit
from .main_1img23 import make_one_image_to_three, LAST_VIEW_PROMPTS
from .for_retry import ensure_image_generated, check_prompt_compliance

PARSED_REPORT_PATH = "parsed_report.json" # main_report.py에서 생성
USER_CHOICE_PATH = "modules/llm_final_api/user_choice.json" # 사용자 선택값 저장
ORG_IMAGE_PATH = "img4new3r_org.png"  # 최종 결과물 이름

def load_json(path: str):
    with open("user_choice.json", "r", encoding="utf-8") as f:
        return json.load(f)



def main_modify_looks():
    validation_detail_prompts: list[str] = []
    
    # ------ 입력 파일/경로 로드 ------
    try:
        parsed_report = load_json(PARSED_REPORT_PATH)
    except Exception as e:
        print(f"parsed_report.json 로드 실패: {e}")
        return

    try:
        user_choice = load_json(USER_CHOICE_PATH)
    except Exception as e:
        print(f"user_choice.json 로드 실패: {e}")
        
    user_query = (user_choice.get("user_choice") or "").strip()
    if not user_query:
        raise ValueError("user_choice.json 에 'user_choice'가 비어 있습니다.")

    # 기준 이미지 결정
    if os.path.exists(ORG_IMAGE_PATH):
        # 이미 수정본이 있는 경우
        base_image_path = ORG_IMAGE_PATH
        print(f"\n기준 이미지: {ORG_IMAGE_PATH} (이전에 생성된 최종본 사용)")
    elif os.path.exists(SELECTED_IMAGE_PATH):
        # 수정본이 없는 경우, main_report에서 선택된 최적 이미지 사용
        base_image_path = SELECTED_IMAGE_PATH
        print(f"\n기준 이미지: {SELECTED_IMAGE_PATH} (최초 선택 이미지 사용)")
    else:
        print("사용할 입력 이미지가 없습니다. SELECTED_IMAGE_PATH 또는 img4new3r_org.png 중 하나는 있어야 합니다.")
        return

    print(f"리포트 파싱 파일: {PARSED_REPORT_PATH}")
    print(f"사용자 선택 파일: {USER_CHOICE_PATH}")

    
    # 현재 이미지 경로 
    current_image_path = base_image_path

    # ------ 사용자 입력대로 편집 실행 ------

    # 자연어 쿼리를 그대로 사용하는 편집 지시문
    edit_instruction = f"""
    사용자 요청을 문장 그대로 충실히 반영하세요.
    요청한 변경 사항을 제외한 사진의 모든 요소는 원본과 완전히 동일하게 유지하세요.

    방의 구조와 카메라 구도는 유지하고, 과도한 재배치나 새로운 가구 추가는 피하세요.
    모든 객체의 배치, 위치, 형태, 크기는 **요청된 변경 대상이 아닌 경우** 그대로 유지해야 합니다.
    각 객체의 정체성이 유지되도록 하되, 재질, 패턴, 시각적 스타일만 자연스럽게 사용자 요청에 맞게 변경하세요.

    단, 실제 사진을 확인했을 때 이미 요청된 상태
    (예: 이미 제거됨, 이미 교체됨, 이미 추가됨)라면
    그 부분은 다시 수정하지 말고 그대로 유지합니다.

    변경하면 안 되는 것 (사용자가 명시적으로 요청한 경우 제외):
    - 배치
    - 가구 개수
    - 객체의 크기나 위치
    - 벽, 바닥, 천장, 창문 구조
    - 조명 방향

    방의 구조, 조명 방향, 텍스처, 기타 가구는
    사용자가 명시적으로 언급하지 않는 이상 변경하지 마세요.
    원근감, 구도, 스케일, 기하 구조도 그대로 유지하세요.

    요청한 변경만 적용하고, 그 외의 모든 요소는 손대지 마세요.

    사용자 요청: "{user_query}"
    """

    # 1차 검수에 넘기기 위해 리스트에 넣어둠
    validation_detail_prompts = [edit_instruction]

    # 실제 이미지 편집 실행
    current_image_path = ensure_image_generated(
        generate_fn=lambda: run_image_edit(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=current_image_path,  # 처음엔 base_image_path
            base_style=None,               # 쓰기 싫으면 None
            edit_instruction=edit_instruction,
            step_name="user",
        ),
        original_path=current_image_path,
    )


    # 자연어 편집 요청이 프롬프트대로 반영되었는지 확인
    print("\n 편집 결과 검수 시작 ---")

    validation_prompt_first = """
    이 이미지는 사용자가 요청한 내용이 자연스럽게 반영되어야 합니다.
    방의 구조와 다른 가구 배치는 유지하면서, 사용자가 언급한 부분만 자연스럽게 변경되었는지 확인하세요.
    """

    ok_first = check_prompt_compliance(
        api_key=API_KEY,
        model_name=REPORT_MODEL,
        user_prompt=validation_prompt_first,
        edited_image_path=current_image_path,
        original_image_path=base_image_path,
        extra_prompts=validation_detail_prompts,
    )
    print(f"[1차 검수 결과] {'PASS' if ok_first else 'FAIL'}")

    if not ok_first:
        print("[1차 검수] FAIL → 전체 편집 한 번 재시도")
        # 방 편집 전체 다시 수행
        current_image_path = base_image_path
        for prompt in validation_detail_prompts:
            current_image_path = ensure_image_generated(
                generate_fn=lambda p=prompt: run_image_edit(
                    api_key=API_KEY,
                    model_name=STYLE_MODEL,
                    input_image_path=current_image_path,
                    base_style=None,
                    edit_instruction=p,
                    step_name="retry",
                ),
                original_path=current_image_path,
            )
        ok_first = check_prompt_compliance(
            api_key=API_KEY,
            model_name=REPORT_MODEL,
            user_prompt=validation_prompt_first,
            edited_image_path=current_image_path,
            original_image_path=base_image_path,
            extra_prompts=validation_detail_prompts,
        )
        print(f"[1차 재검수 결과] {'PASS' if ok_first else 'FAIL'}")

    # ------ 최종 결과물 저장 -------
    final_image_path = current_image_path

    # 최종 결과를 항상 img4new3r_org.png 로 통일
    if os.path.exists(final_image_path) and final_image_path != ORG_IMAGE_PATH:
        save_path = os.path.join('apioutput', ORG_IMAGE_PATH)
        shutil.copyfile(final_image_path, save_path)
    else:
        # 이미 ORG_IMAGE_PATH 를 쓰고 있었던 경우 
        final_image_path = ORG_IMAGE_PATH

    print(f"최종 이미지: {final_image_path}")

    # ------ 좌&우 각도 이미지 생성 ------
    print("\n4단계: 좌&우 각도 이미지 생성")

    try:
        make_one_image_to_three(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=final_image_path,
        )
        print("   - img4new3r_left.png")
        print("   - img4new3r_right.png")
    except Exception as e:
        print(f"4단계(좌/우 각도 생성) 중 에러 발생: {e}")

    # ------ 최종 이미지 프롬프트 준수 검수 ------
    print("\n[2차 검수] 뷰 일관성 검수 시작 ---")

    final_org_path = os.path.join("apioutput", ORG_IMAGE_PATH)
    left_path = os.path.join("apioutput", "img4new3r_left.png")
    right_path = os.path.join("apioutput", "img4new3r_right.png")

    def validate_views():
        ok_left = check_prompt_compliance(
            api_key=API_KEY,
            model_name=REPORT_MODEL,
            user_prompt="카메라가 좌측으로 회전한 동일한 방이어야 합니다.",
            edited_image_path=left_path,
            original_image_path=final_org_path,
            extra_prompts=[LAST_VIEW_PROMPTS.get("left", "")],
        )
        ok_right = check_prompt_compliance(
            api_key=API_KEY,
            model_name=REPORT_MODEL,
            user_prompt="카메라가 우측으로 회전한 동일한 방이어야 합니다.",
            edited_image_path=right_path,
            original_image_path=final_org_path,
            extra_prompts=[LAST_VIEW_PROMPTS.get("right", "")],
        )
        return ok_left and ok_right

    ok_views = validate_views()
    print(f"[2차 검수 결과] {'PASS' if ok_views else 'FAIL'}")

    if not ok_views:
        print("[2차 검수] FAIL → 좌/우 이미지 한 번 재생성 시도")
        make_one_image_to_three(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=final_image_path,
        )
        ok_views = validate_views()
        print(f"[2차 재검수 결과] {'PASS' if ok_views else 'FAIL'}")
