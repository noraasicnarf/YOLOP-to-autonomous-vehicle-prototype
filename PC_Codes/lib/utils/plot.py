## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result):

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    
    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] ==1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)

    color_mask = cv2.resize(color_mask, (img.shape[1],img.shape[0]))
    color_seg = cv2.resize(color_seg, (img.shape[1],img.shape[0]))

    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (640,320), interpolation=cv2.cv2.INTER_CUBIC)

    return img
import cv2
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=None, roi=(115, 253, 349, 319)):
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # Check if the bounding box intersects with the ROI
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    if c2[0] > roi_x1 and c2[1] > roi_y1 and c1[0] < roi_x2 and c1[1] < roi_y2:
        print("Detected")
        
        # Create a separate image for the ROI
        roi_img = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        cv2.imshow("ROI Window", roi_img)

    # Draw bounding box on the original image
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    pass
# def plot():
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#     device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU) if not cfg.DEBUG \
#         else select_device(logger, 'cpu')

#     if args.local_rank != -1:
#         assert torch.cuda.device_count() > args.local_rank
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device('cuda', args.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
#     model = get_net(cfg).to(device)
#     model_file = '/home/zwt/DaChuang/weights/epoch--2.pth'
#     checkpoint = torch.load(model_file)
#     model.load_state_dict(checkpoint['state_dict'])
#     if rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
#     if rank != -1:
#         model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)