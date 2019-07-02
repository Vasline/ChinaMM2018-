import os
import time

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
from get_sample import *
from utils import preprocess_image, EMOTION_DICT

K.set_image_dim_ordering('th')
#tf.python.control_flow_ops = tf



pathDir = '../gen/'
imagesdir = '../testimages/MultPie_train'


#start_flag = True   # 标记，是否是第一帧，若在第一帧需要先初始化
selection = None   # 实时跟踪鼠标的跟踪区域
track_window = None   # 要检测的物体所在区域
drag_start = None   # 标记，是否开始拖动鼠标

def onMouseClicked(event, x, y, flags, param):
    global selection, track_window, drag_start  # 定义全局变量
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        drag_start = (x, y)
        track_window = None
    if drag_start:   # 是否开始拖动鼠标，记录鼠标位置
        xMin = min(x, drag_start[0])
        yMin = min(y, drag_start[1])
        xMax = max(x, drag_start[0])
        yMax = max(y, drag_start[1])
        selection = (xMin, yMin, xMax, yMax)
    if event == cv2.EVENT_LBUTTONUP:   # 鼠标左键松开
        drag_start = None
        track_window = selection
        selection = None




EMOTION_DICT = ["angry", "disgust", "fearful", "happy", "sad", "surprise", "neutral"]
color_list = [(255,0,0),(0,0,255),(255,0,255),(0,255,255)]

SAVED_MODEL_FOLDER_PATH = os.path.abspath('../created_models/30_epoch_training')
SAVED_MODEL_STRUCTURE_FILE_PATH = os.path.join(SAVED_MODEL_FOLDER_PATH, "model_structure.json")
SAVED_MODEL_WEIGHTS_FILE_PATH = os.path.join(SAVED_MODEL_FOLDER_PATH, "model_weights_30_epochs.h5")
#FACE_CASCADE_FILE_PATH = "./face_cascade.xml"
FACE_CASCADE_FILE_PATH = "./haarcascade_frontalface_default.xml"
print("Loading model...")
start_time = time.clock()
with open(SAVED_MODEL_STRUCTURE_FILE_PATH, "r") as f:
    loaded_model_structure = f.read()
model = model_from_json(loaded_model_structure)
model.load_weights(SAVED_MODEL_WEIGHTS_FILE_PATH)
end_time = time.clock()
print("Model is loaded in {0:.2f} seconds".format(end_time - start_time))
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE_PATH)
cv2.namedWindow("camera",0)


#读取摄像头或者Video
cap = cv2.VideoCapture(0)




cv2.setMouseCallback('camera',onMouseClicked)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50))
        if len(faces) == 0:
            print("can't detect faces")
        result = []
        labels = []
        i = 0
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            print("x的值：",x)
            print("y的值：",y)
            print("x+w的值：",x+w)
            print("y+h的值：",y+h)
            cv2.imwrite(pathDir+ "rect"+ str(i)+ ".jpg",face_roi)
            key = cv2.waitKey(1)
            if key == 32:
                start_flag = True
                if start_flag == True:
                    while True:
                        img_first = frame.copy()
                        if track_window:
                            print("开始进行截图")
                            storeImage = img_first[track_window[1]:track_window[3],track_window[0]:track_window[2]]
                            #storeImage = img_first[track_window[0]:track_window[3],track_window[1]:track_window[2]]
                            #storeImage = img_first[track_window[1]:track_window[2],track_window[0]:track_window[3]]
                            print("storeImage的shape",storeImage.shape)
                            print("storeImage的值:",storeImage)
                            print("track_window[0]的值:",track_window[0])
                            print("track_window[1]的值:",track_window[1])
                            print("track_window[2]的值:",track_window[2])
                            print("track_window[3]的值:",track_window[3])
                            #storeImage = cv2.resize(storeImage, (128,128), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(pathDir + "store" + ".jpg",storeImage)
                            cv2.rectangle(img_first, (track_window[0], track_window[1]), (track_window[2], track_window[3]), (0,0,255), 1)
                            print("截图完毕")
                        elif selection:
                            cv2.rectangle(img_first, (selection[0], selection[1]), (selection[2], selection[3]), (0,0,255), 1)
                            cv2.imshow("camera",img_first)
                        if cv2.waitKey(1) == 13:
                            PosImage = cv2.imread(pathDir+"store.jpg")
                            get_sample(PosImage,imagesdir)
                            print("jietu zhongduan")
                            break
                        start_flag = False
                print("退出截图")
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = preprocess_image(face_roi)
            face_roi = np.reshape(face_roi, (1, 1, 48, 48))
            raw_prediction = model.predict_proba(face_roi, verbose=0)[0]
            label = EMOTION_DICT[np.argmax(raw_prediction)]
            labels.append(label)
            print(raw_prediction)
            result.append(raw_prediction)
            #label = EMOTION_DICT[np.argmax(raw_prediction)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            i = i+1
        font = cv2.FONT_HERSHEY_SIMPLEX
        j=0
        for (x1,y1,w1,h1) in faces:
            #cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
            cv2.putText(frame, EMOTION_DICT[0]+":"+str(round(result[j][0],4)), (x1, y1+h1+10), font, 0.4, color_list[j], 1)
            cv2.putText(frame, EMOTION_DICT[1]+":"+str(round(result[j][1],4)), (x1, y1+h1+20), font, 0.4, color_list[j], 1)
            cv2.putText(frame, EMOTION_DICT[2]+":"+str(round(result[j][2],4)), (x1, y1+h1+30), font, 0.4, color_list[j], 1)
            cv2.putText(frame, EMOTION_DICT[3]+":"+str(round(result[j][3],4)), (x1, y1+h1+40), font, 0.4, color_list[j], 1)
            cv2.putText(frame, EMOTION_DICT[4]+":"+str(round(result[j][4],4)), (x1, y1+h1+50), font, 0.4, color_list[j], 1)
            cv2.putText(frame, EMOTION_DICT[5]+":"+str(round(result[j][5],4)), (x1, y1+h1+60), font, 0.4, color_list[j], 1)
            cv2.putText(frame, EMOTION_DICT[6]+":"+str(round(result[j][6],4)), (x1, y1+h1+70), font, 0.4, color_list[j], 1)
            j=j+1
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()






