# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
# from yolov5 import Picture_Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
# from utils.torch_utils import select_device, smart_inference_mode

import os

os.environ[
    "QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/user/anaconda3/envs/ljlyolo/lib/python3.11/site-packages/cv2/qt/plugins"
Picture_Path = ""
save_dir = ""


# Out_Path = ""

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.num = 0
        MainWindow.setStyleSheet("#MainWindow{border-image:url(backgrounds/preview.jpg)}")
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 1200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # MainWindow.setStyleSheet("")

        self.detect_image = QLabel(self.centralwidget)
        self.detect_image.resize(200, 200)
        self.detect_image.move(280, 100)

        self.runs_image = QLabel(self.centralwidget)
        self.runs_image.resize(200, 200)
        self.runs_image.move(480, 100)

        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(500, 10, 500, 41))
        self.label.setStyleSheet("background:transparent")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(300, 80, 111, 31))
        self.pushButton.setObjectName("file")
        self.pushButton.setStyleSheet("color:rgb(0,170,255);\n"
                                      "font: 15pt \"楷体\";\n"
                                      "background-color: rgb(255, 255, 255, 0.3);")
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_2.setGeometry(QtCore.QRect(500, 80, 111, 31))
        self.pushButton_2.setStyleSheet("color:rgb(0,170,255);\n"
                                        "font: 15pt \"楷体\";\n"
                                        "background-color: rgb(255, 255, 255, 0.3);")
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(700, 80, 111, 31))
        self.pushButton_3.setObjectName("file")
        self.pushButton_3.setStyleSheet("color:rgb(0,170,255);\n"
                                        "font: 15pt \"楷体\";\n"
                                        "background-color: rgb(255, 255, 255, 0.3);")
        self.pushButton_3.setObjectName("pushButton_3")


        self.tableWidget = QtWidgets.QTableWidget(MainWindow)
        self.tableWidget.setStyleSheet("background-color: rgba(255, 255, 255, 0.3);")

        # self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setStyleSheet(
            "QHeaderView::section{background-color:rgb(255, 255, 255, 0.3);font:11pt '宋体';color: black;};")
        # self.tableWidget.verticalHeader().setStyleSheet(
        #     "QHeaderView::section{background-color:rgb(255, 255, 255, 0.3);font:11pt '宋体';color: black;};")

        # op = QtWidgets.QGraphicsOpacityEffect()
        # op.setOpacity(0.1)#0.5是透明度
        # self.tableWidget.setGraphicsEffect(op)
        # self.tableWidget.setAutoFillBackground(True)

        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setGeometry(QtCore.QRect(0, 130, 245, 191))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(7)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(80)
        self.tableWidget.verticalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setDefaultSectionSize(15)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 848, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        # print(self.PicturePath)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.msg1)
        self.pushButton_2.clicked.connect(self.msg2)
        # self.pushButton_3.clicked.connect(self.msg3)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow",
                                      "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ff0000;\">YOLOV5目标检测系统</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "选择图片"))
        self.pushButton_2.setText(_translate("MainWindow", "批量选择"))
        # self.pushButton_3.setText(_translate("MainWindow", "检测"))

        # item = self.tableWidget.verticalHeaderItem(0)
        # item.setText(_translate("MainWindow", "1"))
        # item = self.tableWidget.verticalHeaderItem(1)
        # item.setText(_translate("MainWindow", "2"))
        # item = self.tableWidget.verticalHeaderItem(2)
        # item.setText(_translate("MainWindow", "3"))
        # item = self.tableWidget.verticalHeaderItem(3)
        # item.setText(_translate("MainWindow", "4"))
        # item = self.tableWidget.verticalHeaderItem(4)
        # item.setText(_translate("MainWindow", "5"))
        # item = self.tableWidget.verticalHeaderItem(5)
        # item.setText(_translate("MainWindow", "6"))
        # item = self.tableWidget.verticalHeaderItem(6)
        # item.setText(_translate("MainWindow", "7"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "序号"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "置信度"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "个数"))
        self.tableWidget.verticalHeader().setVisible(False)

    def msg1(self, PicturePath):
        # PicturePath, fileType = QtWidgets.QFileDialog.getOpenFileName(None, "选取图片", "./",
        #                                                               "All Files (*);;Text Files (*.txt)")
        PicturePath, fileType = QtWidgets.QFileDialog.getOpenFileName(None, "选取CSV文件", "./", "CSV文件 (*.csv)")
        pix = QPixmap(PicturePath)
        self.detect_image.setPixmap(pix)
        self.detect_image.setScaledContents(True)
        if PicturePath:
            global Picture_Path
            Picture_Path = PicturePath
            # print(Picture_Path)
            opt = parse_opt()
            main(opt)
        for i in range(len(Picture_Path) - 1, 0, -1):
            if (Picture_Path[i] == '/'):
                break
        print(save_dir)
        Out_Path = str(FILE.parents[0]) + "\\" + str(save_dir) + "\\" + Picture_Path[i + 1:len(Picture_Path)]
        print(Out_Path)
        # Out_Path = str(save_dir)
        # print(Out_Path)
        pix = QPixmap(Out_Path)
        self.runs_image.setPixmap(pix)
        self.runs_image.setScaledContents(True)
        newItem = QTableWidgetItem(str(self.num + 1))
        self.tableWidget.setItem(self.num, 0, newItem)
        self.num += 1
        # self.Show()
        # print(self.PicturePath)

    def msg2(self, Filepath):
        VideoPath = QtWidgets.QFileDialog.getExistingDirectory(None, "choose directory", "./")
        pix = QPixmap(VideoPath)
        self.detect_image.setPixmap(pix)
        self.detect_image.setScaledContents(True)
        if VideoPath:
            global Picture_Path
            Picture_Path = VideoPath
            # print(Picture_Path)
            opt = parse_opt()
            main(opt)


# # @smart_inference_mode()
# def run(
#         weights=ROOT / 'yolov5s.pt',  # model path or triton URL
#         source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
#         data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
#         imgsz=(640, 640),  # inference size (height, width)
#         conf_thres=0.25,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         max_det=1000,  # maximum detections per image
#         device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         view_img=False,  # show results
#         save_txt=False,  # save results to *.txt
#         save_conf=False,  # save confidences in --save-txt labels
#         save_crop=False,  # save cropped prediction boxes
#         nosave=False,  # do not save images/videos
#         classes=None,  # filter by class: --class 0, or --class 0 2 3
#         agnostic_nms=False,  # class-agnostic NMS
#         augment=False,  # augmented inference
#         visualize=False,  # visualize features
#         update=False,  # update all models
#         project=ROOT / 'runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         line_thickness=3,  # bounding box thickness (pixels)
#         hide_labels=False,  # hide labels
#         hide_conf=False,  # hide confidences
#         half=False,  # use FP16 half-precision inference
#         dnn=False,  # use OpenCV DNN for ONNX inference
#         vid_stride=1,  # video frame-rate stride
# ):
#     source = str(source)
#     save_img = not nosave and not source.endswith('.txt')  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
#     screenshot = source.lower().startswith('screen')
#     if is_url and is_file:
#         source = check_file(source)  # download
#
#     # Directories
#     global save_dir
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     print(type(save_dir))
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
#
#     # Load model
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#
#     # Dataloader
#     bs = 1  # batch_size
#     if webcam:
#         view_img = check_imshow(warn=True)
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#         bs = len(dataset)
#     elif screenshot:
#         dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     vid_path, vid_writer = [None] * bs, [None] * bs
#
#     # Run inference
#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
#     for path, im, im0s, vid_cap, s in dataset:
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim
#
#         # Inference
#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             pred = model(im, augment=augment, visualize=visualize)
#
#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#
#         # Second-stage classifier (optional)
#         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
#
#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f'{i}: '
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
#
#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # im.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#             s += '%gx%g ' % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, 5].unique():
#                     n = (det[:, 5] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(f'{txt_path}.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                         annotator.box_label(xyxy, label, color=colors(c, True))
#                     if save_crop:
#                         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
#
#             # Stream results
#             im0 = annotator.result()
#             if view_img:
#                 if platform.system() == 'Linux' and p not in windows:
#                     windows.append(p)
#                     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#                     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path[i] != save_path:  # new video
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
#                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer[i].write(im0)
#
#         # Print time (inference-only)
#         LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
#
#     # Print results
#     t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


# def main(opt):
#     check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(**vars(opt))


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
