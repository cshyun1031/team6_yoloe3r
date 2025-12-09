from ultralytics.models.sam import SAM2DynamicInteractivePredictor
from ultralytics import YOLOE
from .crop_by_result import crop_by_result 
import os
import shutil
import torch

def crop(img_path):

    output_root = 'output_crops'
    if os.path.exists(output_root):
        try:
            shutil.rmtree(output_root)
            print(f"🧹 초기화 완료: '{output_root}' 폴더를 비웠습니다.")
        except Exception as e:
            print(f"⚠️ 초기화 실패: {e}")
            
    # ------------------------------------
    # GPU 사용 가능 여부 확인 및 device 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ 추론 장치 설정: {device}")
    
    # SAM2 Predictor 설정
    overrides = dict(conf=0.01, 
                     task="segment", 
                     mode="predict", 
                     imgsz=1024, 
                     model="sam2_t.pt", 
                     save=False,
                     device=device)
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=50)

    # YOLOE 모델 로드 및 GPU로 이동
    model = YOLOE('yoloe-11l-seg.pt')
    if device.startswith("cuda"):
        model.to(device) # 👈 YOLOE 모델을 GPU 메모리로 이동
        print("✅ YOLOE 모델을 GPU로 로드했습니다.")

    names = [
        "Kitchen Cabinet", "Mini Kitchen", "Kitchen Island/Cart", "Kitchen Appliance", 
        "Kitchen Countertop", "Kitchen Pantry", "Kitchen System", "Office Desk/Chair Set", 
        "Conference Chair", "Gaming Furniture", "Conference Table", "Desk/Chair Set", 
        "Office Chair", "Computer Desk", "Vanity Chair/Stool", "Toddler Chair", 
        "Childrens Chair", "Childrens Table", "Step Stool", "Bench", "Cafe Furniture", 
        "Stool", "Bar Table/Chair", "Coffee/Side Table", "Chair", "Table", 
        "Dining Furniture", "Chaise Longue/Couch", "Footstool", "Sofa Bed", 
        "Armchair", "Sofa", "Bedroom Set", "Bed with Mattress", "Bedside Table", 
        "Bed Frame", "Shoe Cabinet", "Storage Unit", "Toy Storage", "Hallway Set", 
        "Partition", "Drawer/Nightstand", "Storage System", "Sideboard/Console Table", 
        "Trolley", "TV/Media Furniture", "Outdoor Storage", "Warehouse Storage", "cabinet"
    ]

    imgs = img_path
    
    # [핵심 1] 모든 프레임의 마스크 데이터를 모을 리스트
    # 구조: [ { "Sofa_0": mask_array, "Table_1": mask_array }, ... ]
    all_frames_masks = [] 
    
    # ID <-> Class Name 매핑 딕셔너리
    id_to_class_map = {}

    for i, img in enumerate(imgs):
        imgsrc = img
        # crop_by_result에서 반환받은 Raw Dict ({0: mask, 1: mask})
        raw_mask_dict = {}

        if i == 0:
            # 1. YOLO 실행 (클래스 식별용)
            model.set_classes(names, model.get_text_pe(names))
            results = model.predict(imgsrc)
            
            # 2. SAM2 메모리 초기화
            # YOLO가 찾은 Box 순서대로 ID(0, 1, 2...)가 부여됨
            predictor(source=imgsrc, 
                      bboxes=results[0].boxes.xyxy.cpu().numpy(), 
                      obj_ids=[k for k in range(len(results[0].boxes))], 
                      update_memory=True)
            
            # 3. [ID 매핑] YOLO 클래스 정보 기록
            if results[0].boxes.cls is not None:
                detected_cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for obj_id, cls_idx in enumerate(detected_cls_ids):
                    raw_name = names[cls_idx] if cls_idx < len(names) else str(cls_idx)
                    
                    # [에러 방지] 이름에 '/'가 있으면 파일시스템 에러 발생 -> '_'로 치환
                    clean_name = raw_name.replace("/", "_") 
                    
                    # ID를 키로 하여 이름 저장 (예: "0" -> "Sofa_0")
                    # 이름 뒤에 ID를 붙여야 중복(Sofa가 2개일 때)을 피할 수 있음
                    id_to_class_map[str(obj_id)] = f"{clean_name}_{obj_id}"

            # 4. 첫 프레임 SAM2 추론 및 저장
            results = predictor(source=imgsrc)
            os.makedirs('output_crops/', exist_ok=True)
            
            # 이때 폴더는 "0", "1"로 생성됨
            raw_mask_dict = crop_by_result(results[0], img)
            
        else:
            # 이후 프레임 SAM2 추적
            results = predictor(source=imgsrc)  
            raw_mask_dict = crop_by_result(results[0], img)

        # [핵심 2] Raw ID를 실제 클래스 이름("Sofa_0")으로 변환하여 리스트에 저장
        current_frame_final_dict = {}
        
        for obj_idx, mask_arr in raw_mask_dict.items():
            str_idx = str(obj_idx)
            # 매핑 정보가 있으면 변환된 이름 사용, 없으면 그냥 ID 사용
            if str_idx in id_to_class_map:
                final_name = id_to_class_map[str_idx] # "Sofa_0"
                current_frame_final_dict[final_name] = mask_arr
            else:
                current_frame_final_dict[str_idx] = mask_arr
        
        all_frames_masks.append(current_frame_final_dict)
        
        if current_frame_final_dict:
            print(f"Frame {i}: Detected {list(current_frame_final_dict.keys())}")


    # ------------------------------------------------------------------
    # [후처리] 폴더 이름 변경 (ID -> ClassName_ID)
    # ------------------------------------------------------------------
    print("\n🔄 폴더 이름을 클래스명으로 변경합니다...")
    
    if os.path.exists(output_root):
        for folder_name in os.listdir(output_root):
            old_path = os.path.join(output_root, folder_name)
            
            # 폴더 이름이 숫자 ID이고, 매핑 정보가 있다면 변경
            if os.path.isdir(old_path) and folder_name in id_to_class_map:
                
                # 이미 위에서 {clean_name}_{obj_id} 형태로 만들어둠
                new_folder_name = id_to_class_map[folder_name] 
                new_path = os.path.join(output_root, new_folder_name)
                
                try:
                    # 중복 방지 (혹시 재실행 시)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"  Changed: '{folder_name}' -> '{new_folder_name}'")
                    else:
                        print(f"  Skip: '{new_folder_name}' already exists.")
                except Exception as e:
                    print(f"  Error renaming {folder_name}: {e}")

    # 최종적으로 [Frame 0: {"Sofa_0": mask...}, Frame 1: ...] 형태 반환
    return all_frames_masks

if __name__ == "__main__":  
    # crop(['image1.jpg', 'image2.jpg'])
    pass