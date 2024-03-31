
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton
from torch.backends import cudnn

from models.experimental import attempt_load

Ui_widget, _ = uic.loadUiType('person_search.ui')
import random

import torch
import torch.nn.functional as F
import os
import sys
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
from PyQt5.QtCore import QTimer, QDateTime
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
ROOT = Path(os.path.abspath(ROOT))  # absolute

model_list = ["resnet18",
              "resnet34",
              'resnet50',
              'se_resnet50',
              'se_resnext50',
              'resnet50_ibn_a',
              ]


# 实现选择camera id的对话框
class Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("select_camera")
        self.camera_id = '0'  # default camera id
        # 设置垂直布局
        layout = QVBoxLayout()
        self.camera_label = QLabel("Select an camera:")
        # 选项框
        self.combo_box_camera_id = QComboBox()
        # 添加选项
        self.combo_box_camera_id.addItems(["0", "1", "2"])
        # 添加按钮
        self.button = QPushButton("OK")

        self.button.clicked.connect(self.on_button_clicked)
        # 上述组件添加至布局中
        layout.addWidget(self.camera_label)
        layout.addWidget(self.combo_box_camera_id)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def on_button_clicked(self):
        self.camera_id = self.combo_box_camera_id.currentText()  # 获取选择的相机 ID
        self.close()


class Dialog_Select_Model(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("select_Model")
        self.model_name = None
        layout = QVBoxLayout()

        self.model_select_label = QLabel("Select_model:")

        self.combo_box_model_name = QComboBox()
        self.combo_box_model_name.addItems(model_list)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.on_button_clicked)
        layout.addWidget(self.model_select_label)
        layout.addWidget(self.combo_box_model_name)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def on_button_clicked(self):
        self.model_name = self.combo_box_model_name.currentText()
        self.close()



class Search_ui(QMainWindow, Ui_widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('person search')
        self.setFixedSize(1268, 867)  # 设置固定的窗口大小
        self.imgz = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.classes = None
        self.vid_stride = 1
        self.dist_thres = 1.5
        self.save_res = True
        self.yolo_model_path = ''
        self.reid_model_path = ''
        self.video_path = ''  # default video path
        self.model_name = 'resnet50_ibn_a'  # default model name
        self.video_show_label_width = self.video_show_label.width()  # get video show label width
        self.video_show_label_height = self.video_show_label.height()  # # get video show label height
        self.load_video_button.clicked.connect(self.load_video)  # set button action
        self.load_query_button.clicked.connect(self.load_query)
        self.load_yolo_model_button.clicked.connect(self.load_yolo_model)
        self.load_reid_model_button.clicked.connect(self.load_reid_model)
        self.select_model_button.clicked.connect(self.open_select_model_dialog)
        self.detect_button.clicked.connect(self.Detect)
        self.camer_button.clicked.connect(self.open_dialog)
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)

        self.dis_str = ''
        self.show()

    def update_time(self):
        self.dateTimeEdit.setDisplayFormat("yyyy年MM月dd日 HH点mm分")
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())


    def Detect(self):
        self.detect(self.video_path, self.imgz, self.yolo_model_path, self.conf_thres,
                    0.45, self.classes, dist_thres=self.dist_thres, save_res=self.save_res)

    def load_yolo_model(self):
        self.model_file_path, _ = QFileDialog.getOpenFileName(self, 'Choose a model file', '', 'Torch Model Files (*.pt *.pth)')
        if self.model_file_path:
            print(self.model_file_path)
            result = 'Model loaded successfully!'
            self.load_model_label.setText(result)
        self.yolo_model_path = self.model_file_path

    def load_reid_model(self):
        self.model_file_path, _ = QFileDialog.getOpenFileName(self, 'Choose a model file', '', 'Torch Model Files (*.pt *.pth)')
        if self.model_file_path:
            print(self.model_file_path)
            result = 'Model loaded successfully!'
            self.load_model_label.setText(result)
        self.reid_model_path = self.model_file_path

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileNames(self, 'Choose a image file', '', 'video file (*.mp4)')
        self.video_path = video_path[0]
        cap = cv2.VideoCapture(video_path[0])
        _, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        first_frame = QImage(np.array(img), np.shape(img)[1], np.shape(img)[0], QImage.Format_RGB888)  # 默认只显示第1帧
        self.video_show_label.setPixmap(QPixmap.fromImage(first_frame))

    def load_query(self):
        query_path = 'query'
        file_list = os.listdir(query_path)
        Qimg = QImage(os.path.join(query_path, file_list[0]))
        pixmap = QPixmap.fromImage(Qimg)
        self.query_image_label.setPixmap(pixmap)
        self.query_image_label.show()

    def detect(self,
               source='0',
               imgsz=640,
               weights='yolov7.pt',
               conf_thres=0.5,
               iou_thres=0.5,
               classes=None,
               dist_thres=1.0,
               save_res=False,
               project=ROOT / 'runs/detect',
               name='exp',
               exist_ok=False,
               show=True):
        source = str(source)
        cap = cv2.VideoCapture(source)

        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = False  # set False for reproducible results

        # ---------- 行人重识别模型初始化 --------------------------
        reidCfg.TEST.WEIGHT = self.reid_model_path
        reidCfg.MODEL.NAME = self.model_name
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
        #print("imgsz :", imgsz)
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

        for path, img, im0s, vid_cap in dataset:
            with torch.no_grad():
                img = torch.from_numpy(img).to(device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
            with torch.no_grad():
                # Inference
                pred = model(img, augment=False)[0]
                # NMS
                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

            # Process predictions
            for i, det in enumerate(pred):  # per image
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
                                crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(
                                    0)  # torch.Size([1, 3, 256, 128])
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
                        distmat = distmat.cpu().detach().numpy()
                        distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果
                        index = distmat.argmin()
                        if distmat[index] < dist_thres:
                            # print('距离：%s' % distmat[index])
                            #self.search_resluts_label.setText('相似度：%s' % distmat[index])
                            self.dis_str = '相似度：%s' % distmat[index]
                            self.search_resluts_label.setText(self.dis_str)
                            plot_one_box(gallery_loc[index], im0, label='find!', color=colors_[int(cls)])

                print('Done.')
                torch.cuda.empty_cache()
                # if webcam:
                #     show = False
                #     cv2.imshow('person search', im0)
                #     if cv2.waitKey(25) == ord('q'):
                #         break

                if show:
                    height, width = im0.shape[:2]
                    ratio1 = width / self.video_show_label_width  # (label 宽度)
                    ratio2 = height / self.video_show_label_height  # (label 高度)
                    ratio = max(ratio1, ratio2)
                    # 格式转换
                    Qim0 = QImage(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), width, height, 3 * width, QImage.Format_RGB888)
                    Qim0 = QPixmap.fromImage(Qim0)
                    # 按照缩放比例自适应 label 显示
                    Qim0.setDevicePixelRatio(ratio)
                    self.video_show_label.setPixmap(Qim0)
                    self.video_show_label.show()
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
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
                                fps, w, h = 25, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

    def open_dialog(self):
        dialog = Dialog(self)
        dialog.exec_()
        self.video_path = dialog.camera_id
        if dialog.camera_id is not None:
            print("Selected camera ID:", self.video_path)


    def open_select_model_dialog(self):
        dialog = Dialog_Select_Model(self)
        dialog.exec_()
        if dialog.model_name is not None:
            self.model_name = dialog.model_name
            print("model name:", self.model_name)
        else:
            raise ValueError("No model name selected")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Search_ui()
    sys.exit(app.exec_())
