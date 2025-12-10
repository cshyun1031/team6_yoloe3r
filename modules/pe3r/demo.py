import math
import copy
import gradio as gr
import os
import torch
import numpy as np
import functools
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
import requests
from io import BytesIO
import cv2
import matplotlib.pyplot as pl
import glob
from copy import deepcopy
import json

# Custom Modules
from modules.pe3r.images import Images
from modules.dust3r.inference import inference
from modules.dust3r.image_pairs import make_pairs
from modules.dust3r.utils.image import load_images, rgb
from modules.dust3r.utils.device import to_numpy
from modules.dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, pts3d_to_trimesh
from modules.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# User API Modules
from modules.llm_final_api.main_report import main_report
from modules.llm_final_api.main_new_looks import main_new_looks
from modules.llm_final_api.main_modify_looks import main_modify_looks
from modules.IR.listup import listup

# -----------------------------------------------------------------------------
# 1. 시각화 및 지오메트리 헬퍼 함수
# -----------------------------------------------------------------------------

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    """
    3D 포인트 클라우드와 카메라 정보를 Trimesh Scene으로 구성하고,
    좌표계를 뷰어에 맞게 변환하여 GLB 파일로 저장하는 함수.
    """
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # 포인트 클라우드 생성
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # 카메라 시각화 추가
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    # 좌표계 변환: 첫 번째 카메라를 기준으로 정렬하고, Y-up 좌표계로 회전 및 Z축 이동
    scene.apply_transform(np.linalg.inv(cams2world[0]))
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    scene.apply_transform(rot)
    translate = np.eye(4)
    translate[2, 3] = -2.0 
    scene.apply_transform(translate)

    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    재구성된 Scene 객체에서 데이터를 추출하고 전처리(하늘 마스킹, 깊이 정리 등)한 후
    GLB 변환 함수를 호출하는 래퍼 함수.
    """
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.ori_imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def mask_nms(masks, threshold=0.8):
    """
    세그멘테이션 마스크 간의 중복을 제거하는 NMS(Non-Maximum Suppression) 함수.
    작은 마스크가 큰 마스크에 일정 비율 이상 포함되면 제거함.
    """
    keep = []
    mask_num = len(masks)
    suppressed = np.zeros((mask_num), dtype=np.int64)
    for i in range(mask_num):
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for j in range(i + 1, mask_num):
            if suppressed[j] == 1:
                continue
            intersection = (masks[i] & masks[j]).sum()
            if min(intersection / masks[i].sum(), intersection / masks[j].sum()) > threshold:
                suppressed[j] = 1
    return keep

def filter(masks, keep):
    """
    NMS 결과 인덱스(keep)에 해당하는 마스크만 남기는 필터링 함수.
    """
    ret = []
    for i, m in enumerate(masks):
        if i in keep: ret.append(m)
    return ret

def mask_to_box(mask):
    """
    이진 마스크로부터 바운딩 박스(x_min, y_min, x_max, y_max) 좌표를 추출하는 함수.
    """
    if mask.sum() == 0:
        return np.array([0, 0, 0, 0])
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    top = np.argmax(rows)
    bottom = len(rows) - 1 - np.argmax(np.flip(rows))
    left = np.argmax(cols)
    right = len(cols) - 1 - np.argmax(np.flip(cols))
    
    return np.array([left, top, right, bottom])

def box_xyxy_to_xywh(box_xyxy):
    """
    [x_min, y_min, x_max, y_max] 포맷의 박스를 [x, y, w, h] 포맷으로 변환하는 함수.
    """
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh

# -----------------------------------------------------------------------------
# 2. YOLO 및 Feature 추출 로직
# -----------------------------------------------------------------------------

def get_class_embedding(class_id, feature_dim=1024):
    """
    클래스 ID를 시드(seed)로 사용하여 고정된 랜덤 임베딩 벡터를 생성하는 함수.
    """
    generator = torch.Generator().manual_seed(int(class_id) + 100) 
    embedding = torch.randn(feature_dim, generator=generator)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding

@torch.no_grad
def get_mask_and_class_from_yolo(seg_model, image_np, original_size, conf=0.25):
    """
    YOLO 모델을 사용하여 이미지에서 마스크와 클래스 ID를 추출하는 함수.
    너무 작은 객체는 필터링하고, NMS를 적용하여 중복을 제거함.
    """
    results = seg_model.predict(image_np, conf=conf, retina_masks=True, verbose=False)
    
    sam_mask = []
    class_ids = []
    img_area = original_size[0] * original_size[1]

    if results[0].masks is not None:
        masks_data = results[0].masks.data
        cls_data = results[0].boxes.cls
        
        for i, mask in enumerate(masks_data):
            bin_mask = mask > 0.5
            if bin_mask.sum() / img_area > 0.002:
                sam_mask.append(bin_mask)
                class_ids.append(int(cls_data[i].item()))

    if len(sam_mask) == 0:
        return [], []

    sam_mask = torch.stack(sam_mask)
    class_ids = torch.tensor(class_ids)

    sorted_idx = torch.argsort(sam_mask.sum(dim=(1, 2)), descending=True)
    sorted_sam_mask = sam_mask[sorted_idx]
    sorted_class_ids = class_ids[sorted_idx]
    
    keep = mask_nms(sorted_sam_mask)
    
    ret_mask = filter(sorted_sam_mask, keep)
    ret_class_ids = [sorted_class_ids[i].item() for i in keep]

    return ret_mask, ret_class_ids


@torch.no_grad
def get_cog_feats(images, pe3r):
    """
    YOLO와 SAM2를 결합하여 다시점 이미지에서 일관된 객체 마스크와 특징 벡터를 추출하는 핵심 로직.
    첫 프레임 YOLO 탐지 -> SAM2 비디오 추적 -> 중간 프레임 YOLO 재탐지 및 병합 과정을 수행함.
    """
    cog_seg_maps = []
    rev_cog_seg_maps = []
    
    inference_state = pe3r.sam2.init_state(images=images.sam2_images, video_height=images.sam2_video_size[0], video_width=images.sam2_video_size[1])
    mask_num = 0
    obj_id_to_class_id = {} 

    np_images = images.np_images
    np_images_size = images.np_images_size
    
    # 첫 프레임 객체 탐지
    masks, class_ids = get_mask_and_class_from_yolo(pe3r.seg_model, np_images[0], np_images_size[0])
    
    for i, mask in enumerate(masks):
        _, _, _ = pe3r.sam2.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=mask_num,
            mask=mask,
        )
        obj_id_to_class_id[mask_num] = class_ids[i]
        mask_num += 1

    video_segments = {} 
    
    # SAM2 비디오 전파 및 추가 객체 탐지
    for out_frame_idx, out_obj_ids, out_mask_logits in pe3r.sam2.propagate_in_video(inference_state):
        sam2_masks = (out_mask_logits > 0.0).squeeze(1)

        video_segments[out_frame_idx] = {
            out_obj_id: sam2_masks[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        if out_frame_idx == 0:
            continue

        yolo_masks, yolo_class_ids = get_mask_and_class_from_yolo(pe3r.seg_model, np_images[out_frame_idx], np_images_size[out_frame_idx])

        for i, yolo_mask in enumerate(yolo_masks):
            flg = 1
            for sam2_mask in sam2_masks:
                area1 = yolo_mask.sum()
                area2 = sam2_mask.sum()
                intersection = (yolo_mask & sam2_mask).sum()
                
                if min(intersection / area1, intersection / area2) > 0.25:
                    flg = 0
                    break
            
            if flg:
                video_segments[out_frame_idx][mask_num] = yolo_mask.cpu().numpy()
                obj_id_to_class_id[mask_num] = yolo_class_ids[i]
                mask_num += 1

    # Feature Vector 생성
    multi_view_clip_feats = torch.zeros((mask_num + 1, 1024))
    
    for obj_id in range(mask_num):
        if obj_id in obj_id_to_class_id:
            cls_id = obj_id_to_class_id[obj_id]
            multi_view_clip_feats[obj_id] = get_class_embedding(cls_id)
        else:
            multi_view_clip_feats[obj_id] = torch.zeros(1024)
            
    multi_view_clip_feats[mask_num] = torch.zeros(1024)

    # 세그멘테이션 맵 생성
    for now_frame in range(len(video_segments)):
        image = np_images[now_frame]
        rev_seg_map = -np.ones(image.shape[:2], dtype=np.int64)
        sorted_dict_items = sorted(video_segments[now_frame].items(), key=lambda x: np.count_nonzero(x[1]), reverse=False)
        for out_obj_id, mask in sorted_dict_items:
            if mask.sum() == 0: continue
            rev_seg_map[mask] = out_obj_id
        rev_cog_seg_maps.append(rev_seg_map)

        seg_map = -np.ones(image.shape[:2], dtype=np.int64)
        sorted_dict_items_rev = sorted(video_segments[now_frame].items(), key=lambda x: np.count_nonzero(x[1]), reverse=True)
        for out_obj_id, mask in sorted_dict_items_rev:
            if mask.sum() == 0: continue
            box = np.int32(box_xyxy_to_xywh(mask_to_box(mask)))
            if box[2] == 0 and box[3] == 0: continue
            
            seg_map[mask] = out_obj_id
            
        cog_seg_maps.append(seg_map)

    return cog_seg_maps, rev_cog_seg_maps, multi_view_clip_feats


# -----------------------------------------------------------------------------
# 3. 핵심 3D 재구성 함수
# -----------------------------------------------------------------------------

def get_reconstructed_scene(outdir, pe3r, device, silent, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    전체 3D 재구성 파이프라인을 실행하는 함수.
    1. 이미지 로드 -> 2. 특징 추출 -> 3. Dust3r 추론 -> 4. 글로벌 정렬 -> 5. 결과 반환
    """
    if len(filelist) < 2:
        raise gr.Error("Please input at least 2 images.")

    images = Images(filelist=filelist, device=device)
    
    try:
        cog_seg_maps, rev_cog_seg_maps, cog_feats = get_cog_feats(images, pe3r)
        imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        rev_cog_seg_maps = []
        for tmp_img in images.np_images:
            rev_seg_map = -np.ones(tmp_img.shape[:2], dtype=np.int64)
            rev_cog_seg_maps.append(rev_seg_map)
        cog_seg_maps = rev_cog_seg_maps
        cog_feats = torch.zeros((1, 1024))
        imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    # 1차 추론 및 정렬
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    
    scene_1 = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats, device=device, mode=mode, verbose=not silent)
    lr = 0.01
    loss = scene_1.compute_global_alignment(tune_flg=True, init='mst', niter=niter, schedule=schedule, lr=lr)

    # 2차 정련(Refinement)
    try:
        import torchvision.transforms as tvf
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in range(len(imgs)):
            imgs[i]['img'] = ImgNorm(scene_1.imgs[i])[None]
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats, device=device, mode=mode, verbose=not silent)
        ori_imgs = scene.ori_imgs
        lr = 0.01
        loss = scene.compute_global_alignment(tune_flg=False, init='mst', niter=niter, schedule=schedule, lr=lr)
    except Exception as e:
        scene = scene_1
        scene.imgs = ori_imgs
        scene.ori_imgs = ori_imgs
        print(f"Refinement failed, using initial scene: {e}")

    # 결과 생성
    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths]) if depths else 1
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs]) if confs else 1
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs


def get_3D_object_from_scene(outdir, pe3r, silent, text, threshold, scene, min_conf_thr, as_pointcloud, 
                             mask_sky, clean_depth, transparent_cams, cam_size):
    """
    (Legacy) 텍스트 프롬프트를 사용하여 YOLO-World로 특정 객체를 검색하고, 
    해당 객체 외 영역을 어둡게 처리하여 3D 모델을 재생성하는 함수.
    """
    if not hasattr(scene, 'backup_imgs'):
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]

    print(f"Searching for: '{text}' using YOLO-World...")

    search_classes = [text] 
    if hasattr(pe3r.seg_model, 'set_classes'):
        pe3r.seg_model.set_classes(search_classes)

    original_images = scene.backup_imgs 
    masked_images = []

    for i, img in enumerate(original_images):
        img_input = img.copy()
        if img_input.dtype != np.uint8:
            if img_input.max() <= 1.0:
                img_input = (img_input * 255).astype(np.uint8)
            else:
                img_input = img_input.astype(np.uint8)

        conf_thr = 0.05 
        
        results = pe3r.seg_model.predict(img_input, conf=conf_thr, retina_masks=True, verbose=False)
        
        combined_mask = np.zeros(img.shape[:2], dtype=bool)
        found = False

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                if mask.shape != combined_mask.shape:
                    mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]))
                combined_mask = np.logical_or(combined_mask, mask > 0.5)
                found = True
        
        if found:
            masked_img = img.copy()
            if img.dtype == np.uint8:
                masked_img[~combined_mask] = 30 
            else:
                masked_img[~combined_mask] = 0.1 
            masked_images.append(masked_img)
        else:
            masked_images.append(img * 0.1)

    scene.ori_imgs = masked_images
    scene.imgs = masked_images 

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)
    return outfile

def highlight_selected_object(
    scene, mask_list, object_id_list,
    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    evt: gr.SelectData,
    outdir=None 
): 
    """
    UI 갤러리에서 선택된 가구 객체를 파란색 틴트로 하이라이트하고 3D 모델을 업데이트하는 함수.
    """
    if scene is None or not mask_list:
        print("⚠️ Scene or mask_list is empty.")
        return None

    if evt is None or not isinstance(evt, gr.SelectData):
        print(f"⚠️ Error: evt is {type(evt)}. Gradio failed to pass SelectData.")
        return None

    selected_index = evt.index
    print(f"🖱️ Clicked index: {selected_index}")

    if selected_index >= len(object_id_list):
        print("Error: Index out of range")
        return None
        
    target_obj_id = object_id_list[selected_index] 
    print(f"🎯 [Highlight] Target Object: {target_obj_id}")

    if not hasattr(scene, 'backup_imgs'):
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]

    masked_images = []
    original_images = scene.backup_imgs
    
    alpha = 0.5 

    for i, img in enumerate(original_images):
        current_frame_masks = mask_list[i]
        
        target_mask = None
        if target_obj_id in current_frame_masks:
            target_mask = current_frame_masks[target_obj_id]
        
        img_h, img_w = img.shape[:2]
        processed_img = img.copy()
        
        if target_mask is not None:
            if target_mask.shape[:2] != (img_h, img_w):
                target_mask = cv2.resize(target_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            if processed_img.dtype == np.uint8:
                roi = processed_img[target_mask].astype(np.float32)
                blue_layer = np.array([0, 0, 255], dtype=np.float32) 
                
                blended = (roi * (1 - alpha)) + (blue_layer * alpha)
                processed_img[target_mask] = blended.astype(np.uint8)
            else:
                roi = processed_img[target_mask]
                blue_layer = np.array([0.0, 0.0, 1.0], dtype=processed_img.dtype)
                
                blended = (roi * (1 - alpha)) + (blue_layer * alpha)
                processed_img[target_mask] = blended
        
        masked_images.append(processed_img)

    scene.ori_imgs = masked_images
    scene.imgs = masked_images

    if outdir is None:
        print("Error: outdir is None")
        return None

    outfile = get_3D_model_from_scene(outdir, False, scene, min_conf_thr, as_pointcloud, mask_sky, 
                                      clean_depth, transparent_cams, cam_size)
    
    return outfile


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    """
    Gradio UI에서 Scene Graph 설정(Swin/Oneref)에 따라 슬라이더 가시성을 조정하는 함수.
    """
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    
    if scenegraph_type == "swin":
        return gr.update(visible=True, value=max_winsize, maximum=max_winsize), gr.update(visible=False)
    elif scenegraph_type == "oneref":
        return gr.update(visible=False), gr.update(visible=True, maximum=num_files - 1)
    else:
        return gr.update(visible=False), gr.update(visible=False)


# -----------------------------------------------------------------------------
# 4. Main Demo UI (Gradio 인터페이스)
# -----------------------------------------------------------------------------

def main_demo(tmpdirname, pe3r, device, server_name, server_port, silent=False):
    
    # Partial Functions 설정
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, pe3r, device, silent)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    get_3D_object_from_scene_fun = functools.partial(get_3D_object_from_scene, tmpdirname, pe3r, silent)

    # 초기 생성 시 3D 모델(상단 변형본, 하단 원본)과 갤러리를 동시에 업데이트하기 위한 래퍼 함수
    def initial_recon_wrapper(*args):
        scene_obj, model_path, gallery_imgs = recon_fun(*args)
        return (
            scene_obj, 
            model_path, # 상단 모델(outmodel)용 경로
            model_path, # 하단 모델(orig_model_display)용 경로 (원본)
            gallery_imgs
        )

    # 스타일 선택 저장 함수
    def save_style_json(selected_style):
        data = {"selected_style": selected_style}
        try:
            with open("modules/llm_final_api/style_choice.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"💾 [Saved] style_choice.json: {data}")
        except Exception as e:
            print(f"❌ [Error] 스타일 저장 실패: {e}")

    # 유저 선택(추가/삭제/변경) 저장 함수
    def save_user_choice_json(use_add, use_remove, use_change):
        data = {
            "use_add": use_add,
            "use_remove": use_remove,
            "use_change": use_change
        }
        try:
            with open("modules/llm_final_api/user_choice.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"💾 [Saved] user_choice.json: {data}")
        except Exception as e:
            print(f"❌ [Error] 유저 선택 저장 실패: {e}")

    # 리포트 파일 읽기 함수
    def read_report_file(filename="report_analysis_result.txt"):
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"파일 읽기 오류: {str(e)}"
        return "⚠️ 분석 결과 파일이 생성되지 않았습니다."

    # 분석 실행 및 UI 업데이트 함수
    def run_analysis_and_show_ui(input_files):
        image_paths = []
        if input_files:
            for f in input_files:
                path = f.name if hasattr(f, 'name') else f
                image_paths.append(path)
        
        if main_report:
            try:
                print(f"📊 [Info] 이미지 분석 시작 ({len(image_paths)}장)...")
                main_report(image_paths) 
            except Exception as e:
                print(f"❌ [Error] 분석 모듈 실행 실패: {e}")
                return f"### 분석 오류 발생\n{str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return "### 분석 모듈 로드 실패", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        report_text = read_report_file("report_analysis_result.txt")
        # 4개의 output 반환 (리포트, 아코디언1, 아코디언2, 버튼 활성화)
        return report_text, gr.update(visible=True, open=True), gr.update(visible=True, open=True), gr.update(visible=True)
    
    # 생성된 이미지 로드 헬퍼 함수
    def load_generated_images(module_func, module_name):
        if module_func:
            try:
                print(f"🎨 [Info] {module_name} 실행 중...")
                module_func()
            except Exception as e:
                print(f"❌ [Error] {module_name} 실패: {e}")
        else:
            print(f"⚠️ Error: {module_name} 모듈이 로드되지 않았습니다.")

        output_dir = os.path.join(os.getcwd(), "apioutput")
        if not os.path.exists(output_dir):
            return []

        files = glob.glob(os.path.join(output_dir, "*.[pP][nN][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][eE][gG]"))
        
        files.sort(key=os.path.getmtime, reverse=True)
        return files[:3]

    def generate_and_load_new_images():
        return load_generated_images(main_new_looks, "main_new_looks")

    def generate_and_load_modified_images():
        return load_generated_images(main_modify_looks, "main_modify_looks")
    
    # 원본 Scene 백업 함수
    def backup_original_scene(scene, input_files):
        saved_paths = []
        if input_files:
            for f in input_files:
                path = f.name if hasattr(f, 'name') else f
                saved_paths.append(path)
        print(f"💾 [Backup] Scene과 파일 {len(saved_paths)}개가 원본으로 백업되었습니다.")
        return scene, saved_paths
    
    # 원본 리포트 백업 함수
    def backup_original_report(report_text):
        print("💾 [Backup] 분석 리포트 텍스트 백업 완료")
        return report_text

    # 원본 복구(Undo) 함수
    def restore_original_scene(orig_scene, orig_inputs, orig_report, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size):
        if orig_scene is None:
            return gr.update(), gr.update(), gr.update(), "⚠️ 저장된 원본이 없습니다."
        
        if hasattr(orig_scene, 'backup_imgs'):
            print("🔄 [Restore] 마스킹된 이미지를 원본으로 복구 중...")
            orig_scene.ori_imgs = [img.copy() for img in orig_scene.backup_imgs]
            orig_scene.imgs = [img.copy() for img in orig_scene.backup_imgs]
        
        restored_model_path = model_from_scene_fun(
            orig_scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size
        )
        
        restored_report = orig_report if orig_report else "🔄 원본 리포트가 없습니다."
        print("↩️ [Restore] 원본 Scene 및 리포트 되돌리기 완료")
        return orig_scene, restored_model_path, orig_inputs, restored_report

    # IR(이미지 검색) 실행 및 갤러리 표시 함수
    def run_and_display(input_files):
        if not input_files: return [], [], []
        
        url_dict, mask_list, ordered_ids = listup(input_files)
        gallery_data = []
        for _, url in url_dict.items():
            try:
                response = requests.get(url[0])
                image = Image.open(BytesIO(response.content))
                gallery_data.append((image, f"Model Name : {url[1]}"))
            except: continue
        return gallery_data, mask_list, ordered_ids
    
    # 갤러리 선택 이벤트 핸들러
    def on_gallery_select(scene, mask_data, id_list, conf, pc, sky, clean, trans, size, evt: gr.SelectData): 
        return highlight_selected_object(scene, mask_data, id_list, conf, pc, sky, clean, trans, size, evt, outdir=tmpdirname)

    # -------------------------------------------------------------------------
    # UI 레이아웃 정의
    # -------------------------------------------------------------------------

    with gr.Blocks(title="IF U Demo", fill_width=True) as demo:

        interior_styles = [
            "AI 추천", "모던 Modern Interior", "미니멀리즘 Minimalist Interior", "스칸디나비아/북유럽 Scandinavian Home",
            "인더스트리얼 Industrial Loft", "클래식 Classic Interior Design", "모던 클래식 Modern Classic Home",
            "빈티지 Vintage Home Decor", "레트로 Retro Style Interior", "내추럴/젠 Natural Zen Interior",
            "재팬디 Japandi Style", "러스틱 Rustic Farmhouse", "팜하우스 Modern Farmhouse",
            "셰비 시크 Shabby Chic Style", "아르데코 Art Deco Design", "미드 센추리 모던 Mid-Century Modern Home",
            "보헤미안/보호 Boho Chic Interior", "트로피컬 Tropical Home Decor", "지중해/스페인 Mediterranean Home",
            "프렌치 French Country Style", "컨템포러리 Contemporary Style", "스팀펑크 Steampunk Decor",
            "고딕 Gothic Interior", "하이테크 Hi-Tech Interior", "그리스 리바이벌 Greek Revival Interior",
            "아르누보 Art Nouveau Interior", "코스탈/해안 Coastal Home Decor", "스위스 샬레 Swiss Chalet Interior",
            "이집트 Egyptian Home Decor", "젠 아시아 Asian Zen Decor", "맥시멀리즘 Maximalist Decor",
            "키치 Kitsch Decor Style", "바이오필릭 Biophilic Design Home", "컬러 블록 Color Block Interior",
            "모노크로매틱 Monochromatic Room", "팝 아트 Pop Art Interior", "그랜디시 Grandmillennial Style",
            "매시큘린 Masculine Interior Design", "페미닌 Feminine Room Decor"
        ]
        
        # State Variables
        scene = gr.State(None)
        original_scene = gr.State(None)        
        original_inputfiles = gr.State(None)
        original_report_text = gr.State(None) 
        mask_data_state = gr.State([])
        object_id_list_state = gr.State([])

        gr.Markdown("## 🧊 IF U Demo")

        with gr.Row():
            # --- 좌측 패널 (입력 및 설정) ---
            with gr.Column(scale=1, min_width=320):
                inputfiles = gr.File(file_count="multiple", label="Input Images")
                
                with gr.Accordion("⚙️ Settings", open=False):
                    schedule = gr.Dropdown(["linear", "cosine"], value='linear', label="schedule")
                    niter = gr.Number(value=300, precision=0, label="num_iterations")
                    scenegraph_type = gr.Dropdown(
                        [("complete", "complete"), ("swin", "swin"), ("oneref", "oneref")],
                        value='complete', label="Scenegraph"
                    )
                    winsize = gr.Slider(value=1, minimum=1, maximum=1, step=1, visible=False)
                    refid = gr.Slider(value=0, minimum=0, maximum=0, step=1, visible=False)
                    min_conf_thr = gr.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20)
                    cam_size = gr.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1)
                    as_pointcloud = gr.Checkbox(value=True, label="As pointcloud")
                    transparent_cams = gr.Checkbox(value=True, label="Transparent cameras")
                    mask_sky = gr.Checkbox(value=False, visible=False)
                    clean_depth = gr.Checkbox(value=True, visible=False)

                run_btn = gr.Button("3D로 변환", variant="primary", elem_classes=["primary-btn"])
                IR_btn = gr.Button("배치된 가구 제품명 찾기", variant="primary", elem_classes=["primary-btn"], visible=False)
                revert_btn = gr.Button("↩️ 원본 되돌리기", variant="secondary")
                
                with gr.Accordion("🎨 분석리포트 적용", open=True, visible=False) as analysis_accordion:
                    add = gr.Checkbox(value=False, label="가구 배치 제안 반영해보기")
                    delete = gr.Checkbox(value=False, label="가구 제거 제안 반영해보기")
                    change = gr.Checkbox(value=False, label="가구 변경 제안 반영해보기")
                    run_suggested_change_btn= gr.Button("결과 생성", variant="primary")
                with gr.Accordion("방 분위기 바꿔보기", open=True, visible=False) as analysis_accordion1:
                    style = gr.Dropdown(interior_styles, value ="AI 추천" , label="style", interactive=True)
                    run_style_change_btn = gr.Button("결과 생성", variant="primary")

                # (Legacy) 숨겨진 텍스트 검색 UI
                with gr.Row(visible=False):
                    text_input = gr.Textbox(label="Query Text")
                    threshold = gr.Slider(label="Threshold", value=0.85)
                    find_btn = gr.Button("Find")

            # --- 우측 패널 (3D 뷰어 및 결과) ---
            with gr.Column(scale=2):
                # 상단 뷰어 (현재 상태)
                outmodel = gr.Model3D(label="3D Reconstruction Result", interactive=True, height="65vh")
                
                # 하단 뷰어 (원본 상태)
                orig_model_display = gr.Model3D(
                    label="Original Model (Reference)", 
                    interactive=True,
                    height="25vh",
                )
                
                analysis_output = gr.Markdown(
                    value="여기에 공간 분석 결과가 표시됩니다.",
                    label="공간 분석 리포트",
                    elem_classes=["report-box"]
                )
                outgallery = gr.Gallery(visible=False)
            
            with gr.Column():
                result_gallery = gr.Gallery(
                    label="Detected Objects", 
                    columns=1, 
                    height="auto", 
                    object_fit="contain" 
                )
                
                IR_btn.click(
                    fn=run_and_display, 
                    inputs=[inputfiles], 
                    outputs=[result_gallery, mask_data_state, object_id_list_state] 
                )

        # ---------------------------------------------------------------------
        # 이벤트 연결
        # ---------------------------------------------------------------------
        
        # 1. 3D 재구성 실행 (최초 실행: 분석 리포트 생성 포함)
        recon_event = run_btn.click(
            fn=initial_recon_wrapper,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, orig_model_display, outgallery]
        )
        
        recon_event.success(
            fn=backup_original_scene,
            inputs=[scene, inputfiles],
            outputs=[original_scene, original_inputfiles]
        ).then(
            fn=lambda: "⏳ 3D 생성이 완료되었습니다. 공간 분위기를 분석 중입니다...", outputs=analysis_output
        ).then(
            fn=run_analysis_and_show_ui,
            inputs=[inputfiles],
            outputs=[analysis_output, analysis_accordion, analysis_accordion1, IR_btn]
        ).success(
            fn=backup_original_report,
            inputs=[analysis_output],
            outputs=[original_report_text]
        )

        # 2. 되돌리기 (Undo)
        revert_btn.click(
            fn=restore_original_scene,
            inputs=[original_scene, original_inputfiles, original_report_text, 
                    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size],
            outputs=[scene, outmodel, inputfiles, analysis_output]
        )

        # 3. 스타일 변경 실행 (분석 리포트 생성 제외)
        run_style_change_btn.click(
            fn=generate_and_load_new_images, inputs=None, outputs=inputfiles
        ).then(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery]
        )
        # [수정됨] run_analysis_and_show_ui 호출 제거됨

        # 4. 제안 적용 (가구 변경 등) 실행 (분석 리포트 생성 제외)
        run_suggested_change_btn.click(
            fn=generate_and_load_modified_images, inputs=None, outputs=inputfiles
        ).then(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery]
        )
        # [수정됨] run_analysis_and_show_ui 호출 제거됨

        # 5. IR 갤러리 선택 인터랙션
        result_gallery.select(
            fn=on_gallery_select,
            inputs=[scene, mask_data_state, object_id_list_state, 
                    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size],
            outputs=outmodel
        )

        # 6. (Legacy) 텍스트 검색 버튼
        find_btn.click(fn=get_3D_object_from_scene_fun,
             inputs=[text_input, threshold, scene, min_conf_thr, as_pointcloud, mask_sky,
                     clean_depth, transparent_cams, cam_size],
             outputs=outmodel)

        # 7. 기타 설정 변경 이벤트
        style.change(fn=save_style_json, inputs=[style], outputs=None)
        add.change(fn=save_user_choice_json, inputs=[add, delete, change], outputs=None)
        delete.change(fn=save_user_choice_json, inputs=[add, delete, change], outputs=None)
        change.change(fn=save_user_choice_json, inputs=[add, delete, change], outputs=None)

        scenegraph_type.change(set_scenegraph_options, [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        inputfiles.change(set_scenegraph_options, [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        
        update_inputs = [scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size]
        for elem in [min_conf_thr, cam_size, as_pointcloud, mask_sky, clean_depth, transparent_cams]:
            if isinstance(elem, gr.Slider): elem.release(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
            else: elem.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)

    demo.launch(share=True, server_name=server_name, server_port=server_port)
