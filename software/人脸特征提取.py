import cv2 as cv
import os

import cv2.face
import numpy as np
from  PIL import Image

#1.自定义一个提取面部特征和标签的函数
def GetImagesandLabels(Path):
    #特征值储存
    facesSamples=[]
    #标签储存
    ids=[]
    #给出所有子文件的路径
    imagePaths=[os.path.join(Path,f) for f in os.listdir(Path)]
    #加载分类器
    face_detector =cv.CascadeClassifier("D:/pycharm/Projs/detector/haarcascade_frontalface_alt.xml")
    #遍历图片
    for imagePath in imagePaths:
        #open是打开函数，也就是将图像文件存放到PIL_img中去，convert表示对图像进行的处理方式，有0（黑白），L(灰度处理),RGB,RGBA……
        PIL_img=Image.open(imagePath).convert('L')
        #将灰度化处理的图片进行向量化，所有灰度值用uint8存放起来
        img_numpy=np.array(PIL_img,'uint8')
        faces=face_detector.detectMultiScale(img_numpy)
        #split函数的作用是将imagePath提取出来以后，先将imagePath分隔开，只取到子文件的名称，split('.')[0]则是将文件名称按照.分开，只取.之前的部分
        id = int(os.path.split(imagePath)[1].split('.')[0])
        #整理提取出的数据
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
        print('id:',id)
        print('faceSample:',facesSamples)
    return facesSamples,ids


if __name__ == '__main__':
#2.储存获取的face和label
    Path='D:\pycharm\Projs\zzy'
    faces,ids=GetImagesandLabels(Path)

#3.加载识别器
    recognizer=	cv.face.LBPHFaceRecognizer.create()
    recognizer.train(faces,np.array(ids))
#4.储存一下
    recognizer.write('trainer.yml')

#接下来学习一下怎么拓展