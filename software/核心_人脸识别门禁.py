import time

import cv2
import cv2 as cv
import datetime
#1.白名单设置，我们用之前的人脸特征提取来将提取获得授权的人脸图像
recognizer=cv.face.LBPHFaceRecognizer.create()
#读取之前训练好的白名单
recognizer.read("D:/pycharm/Projs/train/trainer_zzy.yml")
warntime=0
#这样就做好了分类器了，接下来进行detect函数的编写
def face_detect_cmd(img):
    #1：先将原图片灰度化以后再使用

    img_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    #2:使用已经训练好的分类器(需要额外学习到底怎么制作的分类器)
    face_detect = cv.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    #3：多次识别增加准确度
    face = face_detect.detectMultiScale(img_gray,1.1,5,0,(100,100),(900,900))
    #读取face参数
    find_flag = 0#检测人类靠近
    for x, y, w, h in face:
        # 画框
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=6)
        #画圈
        if (w//2) > (h//2):
            rad=w//2
        else:
            rad=h//2
        cv.circle(img,center=(x+w//2,y+h//2),radius=rad,color=(0,255,0),thickness=6)
    #进行人脸识别,将方框框选的
        #获取时间
        datetime_object = datetime.datetime.now()

        if find_flag==0:
            find_flag=1
            print('在', datetime_object, '有人出现')

        ids, confidence = recognizer.predict(img_gray[y:y+h, x:x+w])
        name = str(ids)
        global warntime  # 声明一个最大识别的次数
        # 显示窗口
        cv.namedWindow('result', cv.WINDOW_NORMAL)
        cv.resizeWindow('result', 780, 560)
        cv.imshow('result', img)
    #划重点，这里的y:y+h,x:x+w实际上是对灰度处理图像画框的位置,predict也是对这一块区域进行的predict
        if confidence > 60:
            warntime += 1
            if warntime > 80:
                #warning()
                #在图片中显示一个未知的标记
                cv2.putText(img,'unknown',(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,0),3)
                print("识别失败！请重试！\n")
                print('在',datetime_object,'识别失败')
                warntime=0
                return 0
        elif confidence <= 60:
            #打印来访人员信息
            cv2.putText(img,name,(x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3)
            print("识别成功!")
            print('在', datetime_object, '识别成功')
            print("来访者是:",name)
            warntime=0
            #cv.destroyAllWindows()马上关闭
            return 1


def recognization():
    #加载摄像头
    cam=cv.VideoCapture(0)
    global flag
    while True:
            flag,frame=cam.read()
            if not flag:
                break
            flag=face_detect_cmd(frame)
            if flag ==0:
                break
            if ord(' ') ==cv.waitKey(1):
                break
            if flag==1:
                break
    cam.release()
    #关闭
    cv.destroyAllWindows()
    return flag

