# 课程1：关于图片的imshow问题
最开始以为是显示的问题<br/>
但是多次调试之后发现，其实是图片尺寸的问题

解决方法参考了以下博客[解决方法](https://blog.csdn.net/threestooegs/article/details/126597738?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169189437616800182175226%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169189437616800182175226&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-126597738-null-null.142^v92^chatsearchT3_1&utm_term=cv2.imshow%E6%98%BE%E7%A4%BA%E5%85%A8%E9%BB%91&spm=1018.2226.3001.4187)<br/>
**最下方的图片尺寸问题的解决方法**

******
******

# 课程2 关于图片灰度转换中的问题 
我遇到的问题是，使用了Imwrite后却保存失败<br/>也就是本地路径上并没有灰度后的图片

怎么使用灰度convert和怎么保存
`cv.imwrite()`是很有用的一个函数<br/>，但是如何让他保存到指定位置呢？

我的初始代码
```
cv.destroyWindow('gray_dzr')
cv.imwrite('./gray_dzr.jpg',gray_img)
```
结果将关闭特定窗口改为关闭所有窗口以后竟然就可以正常打开关闭了，太离谱
```
修改后的代码如下
cv.destroyAllWindows()
cv.imwrite('./gray_dzr.jpg',gray_img)
```
然后就可以正常保存了

```
捏马，竟然不改这个也可以正常允许，神魔Bug
```

## 一个小小idea
如果有多个图片，那么怎么实现多个文件的命名呢

我的代码如下：

```
for num in range(1,10):
    filename ='gray_dzr'+str(num)+'.jpg'
    file_exist=open(filename,mode='r')
    if not file_exist:
        cv.imwrite(filename,gray_img)
        file_exist.close()
    num += 1
    break
else:
        print(str(filename))
```
但是发现了许多问题，所以我使用chatgpt帮助我更正了一下
```
import os
import cv2 as cv

gray_img = ...  # Assuming you have your gray image

for num in range(1, 11):
    filename = 'gray_dzr' + str(num) + '.jpg'
    
    if os.path.exists(filename):
        print(f"File '{filename}' already exists.")
    else:
        cv.imwrite(filename, gray_img)
        print(f"Created file '{filename}'.")  
    
    if num >= 9:  # To limit the loop to 1-10
        break

```
非常优美的解决方案，学习到了`print(f"a real string '{名称}' 写法")`
<br/>也就是打印标准字符串
还学到了os的包操作检查文件是否存在

拓展一下的话：可以想一下怎么去实现自动的：<br/>新建一个图片文件夹->放入图片


***
# 3.图片裁剪
`cv.resize(img,dsize=(length,width))`可以实现对图片尺寸的裁剪<br/>
`ord('任意键')`可以用来检测是否按到某一个键

# 4.画框（没啥好说的）

# 5.识别人脸
重头戏！<br/>我觉得我没学明白 
<br/>总之需要自己定义一个人脸识别函数出来
<br/>但是所需要的分类器则实际上是opencv预设好的了，如果需要拓展的话就要学习一下怎么去自己设置分类器啥的
<br/>发现**问题一**，在使用opencv自带的分类器时，不能将带有中文的路径带入进去，不然会出现打不开文件的情况

重点在思路：
```
先二值化（灰度处理），再进行分类器识别，最后复筛几次，获取参数，然后原图像画框
```

```
def face_detect_cmd():
    #1：先将原图片灰度化以后再使用
    img_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    cv.namedWindow('zzy', cv.WINDOW_NORMAL)
    cv.resizeWindow('zzy', 1000, 1000)
    cv.imshow('zzy', img_gray)
    cv.waitKey(0)
    #2:使用已经训练好的分类器(需要额外学习到底怎么制作的分类器)
    face_detect = cv.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    #3：多次识别增加准确度
    face = face_detect.detectMultiScale(img_gray,1.01,5,0,(100,100),(900,900))
    #读取face参数
    for x, y, w, h in face:
        # 画框
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=6)
    cv.namedWindow('zzy', cv.WINDOW_NORMAL)
    cv.resizeWindow('zzy', 1000, 1000)
    cv.imshow('zzy', img)
```
需要注意的是，这个分类器一次性可以识别同一张照片中的多个脸并框选出来，如果分类效果不够理想的话可以对分类器进行重选

# 打开摄像头咯
关键代码很简单
```
#摄像头打开
cam=cv.VideoCapture(0)
#进行人脸识别
while True:
    flag, frame = cam.read()
    if not flag:
        break
    face_detect_cmd(frame)
    if ord('q')==cv.waitKey(0):
        break

cv.destroyAllWindows()
cam.release()
```
注意，VideoCapture()括号中的可以填视频文件，这样可以实现对指定视频的捕获
但是一定不要忘了cam.release()，这一步释放很重要
<br/>如果是VideoCapture(0)，那就是调用系统摄像头
<br/>但是有一个问题，那就是调用摄像头以后他不能自己刷新，需要我按空格它才刷新，不知道该怎么解决


<br/>**原来是Waitkey的锅**，只需要将waitkey的时间缩短就可以了，比如waitkey(1)
<br/>此外需要注意，read函数返回的是一个二元组，第一个值是一个bool值，第二个值是图像对象
所以`flag, frame = cam.read()`成对出现才是正确的写法


# 6.人脸的识别

所需要用到的工具有：numpy,PIL(图像处理工具库)
<br/>这是一个有一丢丢复杂的过程
`找到训练文件夹->识别并提取出训练文件夹中图片的标签和特征->训练->存放`
我们依次来理解一下
<br/>首先是训练文件夹，我们先在纯英文路径下存放一个图片文件夹，里面都是我们要进行特征标注的文件
<br/>其次就是提取过程了,我们用faces和ids来存放这些图片对应的面部特征和ids
<br/>那么具体是怎么提取的呢，这个需要我们自己来编写对应的函数，具体过程如下：
```
def GetImagesandLabels(Path):
    #特征值储存
    facesSamples=[]
    #标签储存
    ids=[]
    #给出所有子文件的路径
    imagePaths=[os.path.join(Path,f) for f in os.listdir(Path)]
    #加载分类器
    face_detector =cv.CascadeClassifier("D:/Edge下载/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")
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
        print('id:',ids)
        print('faceSample:',facesSamples)
    return facesSamples,ids
```
`@Path->faces,ids`这个函数就是一个将对应路径文件下所有的图片全部都识别出特征值并将id提取的函数

`os.path.join(path,name)`效果就是得出path\name的一个地址的衔接效果，更好的表示地址，os.listdir(path)就是将path目录下的所有的文件名称形成一个list
<br/>分类器就不用赘述了，实现对人脸的检测

接下来对所有的face进行遍历，对每一个face都先进行灰度化处理，将图片转为向量，然后将向量传输到numpy矩阵中去
<br/>这折后就是将id提取出来,id就是文件名称而已.
<br/>最后将所有的id和facesample提取出来汇总，这样就得到了每一个图片的id和特征矩阵


`训练`：就是加载识别器，将我们指定的脸的特征值通过一定的方式训练后存放到一个文件中去
```
#3.加载识别器
    recognizer=cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
```
这里有一个小插曲,cv.face这个对象必须是要下载一个特定的包才能使用，否则无法使用的
参考链接：[解决cv.face无法使用的问题](https://blog.csdn.net/m0_45045158/article/details/124310682?ops_request_misc=&request_id=&biz_id=102&utm_term=lbphfacerecognizer%E6%97%A0%E6%B3%95%E8%A7%A3%E6%9E%90&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-124310682.142^v92^chatsearchT3_1&spm=1018.2226.3001.4187)
### **重大挫折！**
在使用识别器加载过程中debug是十分艰辛的，尤其是在前面下载完了看例程，我还以为可以一模一样的写就完了，结果发现opencv的4.8版本过高，之前的`cv.face.LBPHFaceRecognizer_create()`函数根本无法使用，经过了漫长探索才发现，原来是初始化这个class的变量方法变了，更改以后才可以用
<br/>一个参考文章，对我很有启发，那就是查官方手册
<br/>文章源地址[怎么解决LBPHFaceRecognizer_create()不存在的问题](https://blog.csdn.net/qq_45788060/article/details/129465724?ops_request_misc=&request_id=&biz_id=102&utm_term=opencv%E4%B8%AD%E6%89%BE%E4%B8%8D%E5%88%B0LBPHFaceRecognizer&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-129465724.142^v92^chatsearchT3_1&spm=1018.2226.3001.4187)

```
代码如下：
#3.加载识别器
    recognizer=	cv.face.LBPHFaceRecognizer.create()
    recognizer.train(faces,np.array(ids))
```
竟然是类对象里的方法！捏麻麻地


*`存放`*:将训练的结果通过yml格式的文件存放下来，yml简而言之就是一种用于存放数据和标签对应的文件
<br/>本来是希望能够写一个直接新建文件夹的过程，但是发现直接这么写要出bug，所以就采用了直接write文件的方法，至于生成train文件夹的方法，我想先留白，以后再说咯
<br/>主要是看别人也开摆，我也想摆了(乐) [存放问题](https://blog.csdn.net/CSDN_Yangk/article/details/127259940)


# 7.做点有意思的应用！（比如智能门禁）
本来不知道怎么做的，但是突然想起来以前还做过一个python自动发送邮件的项目，感觉可以联系起来


思路大概是这样的:
```
核心:人脸识别
工作方式：
1.先将所有允许的人员录入到名单文件夹中
2.红外线传感器，检测外部是否有来往人员
3.打开摄像头：人脸检测
            合法人脸：打开舵机开门
            非法人脸：继续检测（并统计检测次数），超过最大次数判定为非法人员，亮LED_RED，禁止开门。
4.陌生人报警模式：如果红外检测长时间检测到有异物存在，打开摄像头，并进行摄像头截图，将截图内容保存，通过短信方式发送给guard，可以增加一个外放语音模块，对外物进行驱离。
```
存在几个难题：
<br/>1.对警报程序的编写
<br/>2.人脸识别的编写
<br/>3.自动截图程序的编写
<br/>4.上位机和单片机之间通信的问题
<br/>5.舵机的操作
<br/>6.红外传感器的检测
<br/>7.摄像头的操作
<br/>8.控制LED屏幕显示警告

对于不知道怎么通信的过程而言，我现在的解决办法是
<br/>1.通过python来写一下串口通信
<br/>2.单片机的烧录参考野火的例程
<br/>比如串口通信过程
<br/>舵机操作就参考江科大的例程
<br/>红外传感器的话还没想好，先搁置
<br/>LED还没学，学了再说


优先级：

先实现人脸识别的软件过程
<br/>再实现上位机和单片机的互相通信（基于python）
<br/>再实现对舵机的控制
<br/>再实现对红外传感器的控制
<br/>可以加一个蜂鸣器或者喇叭控制
<br/>再实现对LED的操作


接下来的过程就交给开发文档吧！

# 长远计划栏
<br/>1.小型化+模块化，实现不需要上位机也可以实现的系统？
<br/>2.如果不用python呢？
<br/>3.语音啥的能不能安排上


# 现在看来还有很多要学习的地方捏