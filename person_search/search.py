import argparse
import random
import torch
import torch.nn.functional as F
import os
import sys

from torch.backends import cudnn

from models.experimental import attempt_load

sys.path.append('.')
from reid.data.transforms import build_transforms
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, increment_path,scale_coords
from utils.plots import plot_one_box
from reid.data import make_data_loader
from pathlib import Path
from reid.modeling import build_model
from reid.config import cfg as reidCfg
import numpy as np
from PIL import Image
import cv2
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def detect(
           source='0',
           imgsz=(640, 640),
           weights='yolov7.pt',
           half=False,
           dist_thres=1.0,
           save_res=False,
           project='runs/detect',
           name='exp',
           exist_ok=False):
    source = str(source)
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False  # set False for reproducible results


    # ---------- 行人重识别模型初始化 --------------------------
    query_loader, num_query = make_data_loader(reidCfg)  # 验证集预处理
    reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
    reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
    reidModel.to(device).eval()  # 模型测试

    query_feats = []  # 测试特征
    query_pids = []  # 测试ID

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch  # 返回图片，ID，相机ID
            img = img.to(device)  # 将图片放入gpu
            feat = reidModel(img)  # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            query_feats.append(feat)  # 获得特征值列表
            query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = F.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量

    # --------------- yolov7 行人检测模型初始化 -------------------
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    #print("imgsz:", imgsz)
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors_ = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        # NMS
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=False)
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == 'person':
                        print('%g %ss' % (n, names[int(c)]), end=', ')  # 打印个数和类别

                gallery_img = []
                gallery_loc = []  # 这个列表用来存放框的坐标
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == 'person':
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        w = xmax - xmin
                        h = ymax - ymin
                        # 如果检测到的行人太小了，感觉意义也不大
                        # 这里需要根据实际情况稍微设置下
                        if w * h > 500:
                            gallery_loc.append((xmin, ymin, xmax, ymax))
                            crop_img = im0[ymin:ymax,
                                       xmin:xmax]  # HWC (602, 233, 3)  这个im0是读取的帧，获取该帧中框的位置 im0= <class 'numpy.ndarray'>

                            crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                            crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                            gallery_img.append(crop_img)
                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reidModel(gallery_img)  # torch.Size([7, 2048])
                    print("The gallery feature is normalized")
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                    m, n = query_feats.shape[0], gallery_feats.shape[0]
                    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                              torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()

                    distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                    distmat = distmat.cpu().numpy()
                    distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果
                    index = distmat.argmin()
                    if distmat[index] < dist_thres:
                        # print('距离：%s' % distmat[index])
                        plot_one_box(gallery_loc[index], im0, label='find!', color=colors_[int(cls)])

            print('Done.')
            torch.cuda.empty_cache()
            if webcam:
                cv2.imshow('person search', im0)
                cv2.waitKey(25)
            if save_res:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='person search')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='demo.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640,
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dist_thres', type=float, default=1.5, help='dist_thres')
    parser.add_argument('--save_res', action='store_true', default=True, help='save detection results')

    opt = parser.parse_args()
    logger.info(opt)
    weights, source, imgsz, half, dist_thres, save_res = opt.weights, opt.source, opt.imgsz,  opt.half, opt.dist_thres, opt.save_res

    with torch.no_grad():
        detect(source, imgsz, weights, half, dist_thres=dist_thres, save_res=save_res)
