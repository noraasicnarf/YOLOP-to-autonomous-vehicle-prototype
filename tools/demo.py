import argparse
import os, sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torchvision.transforms as transforms
from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.core.function import AverageMeter
from lib.core.postprocess import connect_lane
from lib.dataset.convert import id_dict
from tqdm import tqdm
from tools.autodrive import AutoDriveLaneDetection

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(opt.source, img_size=opt.img_size)

    #Autonomous Driving Class
    autodrive = AutoDriveLaneDetection(opt.ip, opt.udp_port)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    id_rev_dict = {value: key for key, value in id_dict.items()}


    # Run inference
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    for _, (img, img_det, shapes) in tqdm(enumerate(dataset),total = len(dataset)):

        
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        
        
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        # Display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

    # Display FPS on the upper right corner
        cv2.putText(img_det, f'FPS: {fps:.2f}', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        _, _, height, width = img.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

    #Drivable area Processing
        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        image_height = da_seg_mask.shape[0]

        # Create a binary mask for the 1/3 bottom part crosswise
        roi_mask = np.zeros_like(da_seg_mask)
        roi_mask[image_height * 2 // 3:, :] = 1

        # Apply the mask to the original image or ll_seg_mask
        da_seg_mask = da_seg_mask * roi_mask

    #Lane Processing
        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = autodrive.process_lines(ll_seg_mask)
        ll_seg_mask = connect_lane(ll_seg_mask)

        image_height, image_width = ll_seg_mask.shape

        # Create a binary mask for the 1/3 bottom part crosswise
        roi_mask = np.zeros_like(ll_seg_mask)
        roi_mask[image_height * 3 // 4:, :] = 1

        # Apply the mask to the original image or ll_seg_mask
        ll_seg_mask = ll_seg_mask * roi_mask
        
        #Autonomous Driving
        img_det = autodrive.auto_drive(img_det,(da_seg_mask,ll_seg_mask))


        #Object Detection
        cv2.rectangle(img_det,(125, 275),(355, 320), [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                target_value = int(names[int(cls)])
                category = id_rev_dict.get(target_value)
                autodrive.plot_one_box(xyxy, img_det , label=category, color=colors[int(cls)], line_thickness=2)
        
        cv2.imshow('image', img_det)
        cv2.waitKey(1)  # 1 millisecond


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='weights/epoch-140.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')

    # New arguments
    parser.add_argument('--ip', type=str, default='192.168.1.3', help='IP address')
    parser.add_argument('--camport', type=int, default=5555, help='Camera port')
    parser.add_argument('--udp_port', type=int, default=8889, help='UDP port')

    opt = parser.parse_args()

    
    with torch.no_grad():
        detect(cfg,opt)
