import cv2
import matplotlib.pyplot as plt
import numpy as np


def imgAttribute(imgPath):
    img = cv2.imread(imgPath)
    # 图片的h长，w宽，c通道数（3 == RGB）
    print(img.shape)


def imgRW(windowName: str, imgPath: str):
    # 读取图片（图片的路径）,默认为彩色(c == 3)
    img_c = cv2.imread(imgPath, 3)  # img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
    # 读取图片（图片的路径）,灰度(c == 0)
    img_g = cv2.imread(imgPath, 0)

    # 在窗口显示图片（窗口名，读取图片的变量）
    cv2.imshow(windowName, img_c)
    # 等待1000ms
    cv2.waitKey(1000)

    cv2.imshow(windowName, img_g)
    cv2.waitKey(1000)

    # jpg格式保存图片
    cv2.imwrite("images/apple_gray.jpg", img_g)

    # 图片的底层数据实现
    print(type(img_c))  # <class 'numpy.ndarray'>

    print(img_c.size)  # 图片的像素个数

    print(img_c.dtype)  # 图片的像素类型


# 边界填充
def fillBoundary(img):
    top_size, bottom_size, left_size, right_size = (
        100,
        100,
        100,
        100,
    )  # 填充的边界大小

    replicate = cv2.copyMakeBorder(
        img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE
    )  # 复制边界
    reflect = cv2.copyMakeBorder(
        img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT
    )  # 镜像边界
    reflect101 = cv2.copyMakeBorder(
        img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101
    )  # 镜像边界
    wrap = cv2.copyMakeBorder(
        img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP
    )  # 循环边界
    constant = cv2.copyMakeBorder(
        img,
        top_size,
        bottom_size,
        left_size,
        right_size,
        cv2.BORDER_CONSTANT,
        value=(0, 33, 255),
    )  # 常数边界

    imgs = [img, replicate, reflect, reflect101, wrap, constant]

    for i in range(6):
        cv2.imshow("replicate", imgs[i])
        cv2.waitKey(1000)


# 数值计算
def imgMath(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        img_plus = cv2.add(img, img)[:, :, :]  # 给每个像素（bgr）加10
        cv2.imshow("img_plus", img_plus)
        cv2.waitKey(000)


# 图像融合 相同尺寸、相同通道数的图像才能进行融合
def imgFusion(img1, img2):
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))

    img1_small = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)  # 缩小图片
    cv2.imshow("img1", img1_small)
    cv2.waitKey(000)

    cv2.imshow("img1", img1)
    cv2.waitKey(000)
    cv2.imshow("img2", img2)
    cv2.waitKey(000)

    img_add = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)  # 图像融合
    cv2.imshow(img_add)
    # plt.waitKey(000)


# 图像阈值
def imgThreshold(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,dst = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY) # 二值化,只能灰度图
    # ret,dst = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV) # 反二值化,只能灰度图
    # ret,dst = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC) # 截断,只能灰度图
    # ret,dst = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO) # 阈值以下的设为0,只能灰度图
    ret, dst = cv2.threshold(
        img_gray, 127, 255, cv2.THRESH_TOZERO_INV
    )  # 阈值以下的设为0,反转,只能灰度图
    if ret:
        cv2.imshow("dst", dst)
        cv2.waitKey(000)


# 图像滤波
def imgFilter(img):
    img = cv2.resize(img, (300, 300))

    cv2.imshow("img", img)
    cv2.waitKey(000)

    blur = cv2.blur(img, (5, 5))  # 均值模糊
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
    median = cv2.medianBlur(img, 5)  # 中值模糊
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波

    res = np.hstack((img, blur, gaussian, median, bilateral))  # 合并图片

    cv2.imshow("res", res)
    cv2.waitKey(000)


# 腐蚀操作
def imgErode(img):
    kernel = np.zeros((3, 3), np.uint8)  # 定义核
    erosion = cv2.erode(img, kernel, iterations=20)  # 腐蚀操作
    cv2.imshow("erosion", erosion)
    cv2.waitKey(000)


# 膨胀操作
def imgDilate(img):
    kernel = np.ones((3, 3), np.uint8)  # 定义核
    dilation = cv2.dilate(img, kernel, iterations=1)  # 膨胀操作
    cv2.imshow("dilation", dilation)
    cv2.waitKey(000)


# 开运算
def imgOpen(img):
    kernel = np.ones((3, 3), np.uint8)  # 定义核
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    cv2.imshow("opening", opening)
    cv2.waitKey(000)


# 闭运算
def imgClose(img):
    kernel = np.ones((3, 3), np.uint8)  # 定义核
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
    cv2.imshow("closing", closing)
    cv2.waitKey(000)


# 图像梯度 膨胀 - 腐蚀（图像运输）
def imgGradient(img):
    kernel = np.ones((3, 3), np.uint8)  # 定义核
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # 图像梯度
    cv2.imshow("gradient", gradient)
    cv2.waitKey(000)


# 礼帽黑帽操作
"""礼帽（Top Hat）和黑帽（Black Hat）操作是形态学图像处理中的两种重要技术，用于对图像进行特定的形态学变换。

礼帽操作（Top Hat）
礼帽操作通常用于提取图像中小的亮区域。它是原始图像与其开运算（morphological opening）结果之间的差值。开运算是先进行腐蚀操作后进行膨胀操作。

公式： 

应用：

礼帽操作可以帮助突出一些细小的、亮的特征，适合用于背景均匀但存在小亮点的图像中。
黑帽操作（Black Hat）
黑帽操作与礼帽操作类似，但它用于提取图像中小的暗区域。它是闭运算（morphological closing）结果与原始图像之间的差值。闭运算是先进行膨胀操作后进行腐蚀操作。

公式： 

应用：

黑帽操作可以帮助突出一些细小的、暗的特征，适合用于背景均匀但存在小黑点的图像中。
总结
礼帽操作：用于提取小的亮区域。
黑帽操作：用于提取小的暗区域。
两者都是通过将图像与其形态学操作结果相减来实现的，适用于多种图像处理应用，如特征提取和图像分割等。"""


def imgCanny(img):
    kernel = np.ones((3, 3), np.uint8)  # 定义核
    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)  # 礼帽操作
    img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)  # 黑帽操作
    res = np.hstack((img, img_tophat, img_blackhat))  # 合并图片
    cv2.imshow("res", res)
    cv2.waitKey(000)


# 图像梯度Sobel算子 边缘检测
def imgSobel(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img', img)
    # cv2.waitKey(000)
    # x方向求导
    sobel_x = cv2.Sobel(
        img, cv2.CV_64F, 1, 0, ksize=3
    )  # 3x3 Sobel算子 水平方向 参数列表：图像，数据类型，x方向，y方向，卷积核大小
    # cv2.imshow('img', sobel_x)
    # cv2.waitKey(000)
    sobel_x = cv2.convertScaleAbs(
        sobel_x
    )  # 转换为uint8类型 ,因为cv2.Sobel()函数返回的结果是float32类型
    # cv2.imshow('img', sobel_x)
    # cv2.waitKey(000)
    # y方向求导
    sobel_y = cv2.Sobel(
        img, cv2.CV_64F, 0, 1, ksize=3
    )  # 3x3 Sobel算子 垂直方向 参数列表：图像，数据类型，x方向，y方向，卷积核大小
    # cv2.imshow('img', sobel_y)
    # cv2.waitKey(000)
    sobel_y = cv2.convertScaleAbs(
        sobel_y
    )  # 转换为uint8类型,因为cv2.Sobel()函数返回的结果是float32类型
    # cv2.imshow('img', sobel_y)
    # cv2.waitKey(000)
    # 合并图片
    # res = np.hstack((img, sobel_x, sobel_y))

    sobelxy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)  # 图像融合
    # cv2.imshow('sobelxy', sobelxy)
    # cv2.waitKey(000)

    ret = np.hstack((img, sobel_x, sobel_y, sobelxy))  # 合并图片
    cv2.imshow("ret", ret)
    cv2.waitKey(000)


# 图像梯度Scharr算子 边缘检测
def imgScharr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # x方向求导
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # 3x3 Scharr算子 水平方向
    scharr_x = cv2.convertScaleAbs(
        scharr_x
    )  # 转换为uint8类型 ,因为cv2.Sobel()函数返回的结果是float32类型
    # y方向求导
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # 3x3 Scharr算子 垂直方向
    scharr_y = cv2.convertScaleAbs(
        scharr_y
    )  # 转换为uint8类型,因为cv2.Sobel()函数返回的结果是float32类型
    # 合并图片
    res = np.hstack((img, scharr_x, scharr_y))
    cv2.imshow("res", res)
    cv2.waitKey(000)


# 图像梯度Laplacian算子 边缘检测
def imgLaplacian(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 拉普拉斯算子
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)  # 转换为uint8类型
    cv2.imshow("laplacian", laplacian)
    cv2.waitKey(000)


def ThreeS(img):
    img = cv2.resize(img, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # 3x3 Scharr算子 水平方向
    scharr_x = cv2.convertScaleAbs(
        scharr_x
    )  # 转换为uint8类型 ,因为cv2.Sobel()函数返回的结果是float32类型
    # y方向求导
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # 3x3 Scharr算子 垂直方向
    scharr_y = cv2.convertScaleAbs(
        scharr_y
    )  # 转换为uint8类型,因为cv2.Sobel()函数返回的结果是float32类型

    scharr = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    ret = np.hstack((img, sobel, scharr, laplacian))  # 合并图片
    cv2.imshow("ret", ret)
    cv2.waitKey(000)


# 边缘检测
def imgEdge(img):
    img_t = img
    # img_t = cv2.resize(img_t, (500, 900))
    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    img_blur = cv2.GaussianBlur(img_t, (5, 5), 0)
    # 边缘检测
    canny_1 = cv2.Canny(img_blur, 70, 75)  # 参数列表：图像，低阈值，高阈值
    return canny_1


# 图像金字塔
def imgPyramid(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像金字塔
    pyramid = [img_gray]
    for i in range(6):
        img_gray = cv2.pyrDown(img_gray)
        pyramid.append(img_gray)
    # 合并图片
    for i in range(6):
        cv2.imshow("img_pyramid_" + str(i), pyramid[i])
    cv2.waitKey(0)


def videoRW(windowName: str, videoPath: str):
    # 读取视频（视频的路径）
    cap = cv2.VideoCapture(videoPath)

    # 判断视频是否打开成功
    if cap.isOpened():
        # 读取视频的第一帧
        """
        cap.read() 函数返回一个元组，包含两个元素。
        通过 opened, frame = cap.read() 的语法，
        我们将这个元组的第一个元素赋值给变量 opened,第二个元素赋值给变量 frame。
        """
        opened, frame = cap.read()  # 读取视频的第一帧
    ii = 1
    count = 1
    while opened:
        ret, frame = (
            cap.read()
        )  # 读取视频的每一帧,ret表示是否读取成功，frame表示读取的帧
        if frame is None:
            break
        if ret is True:
            cur_img = frame.copy()  # 复制当前帧
            """
            img数据结构：
            img[y][x][c]
            y表示行，x表示列，c表示通道
            其中c的取值范围为0-2，分别表示BGR
            xyc , xy 表示坐标的位置，c表示颜色通道 ，可以选择需要修改的局部区域
            c 中也是一个二维数组，分别表示BGR[x,y] = colorValue
            """
            cur_img = imgEdge(cur_img)  # 边缘检测
            cur_img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            ret = np.hstack((cur_img_gray, cur_img))  # 合并图片
            all = cv2.addWeighted(cur_img_gray, 1.0, cur_img, 1.0, 0)  # 图像融合
            cv2.imshow(windowName, all)  # 在窗口显示当前帧
            if count % 5 == 0:
                count += 1
                cv2.imwrite(f"vidoes_imgs/count{ii}.jpg", all)
                cv2.imwrite(f"vidoes_imgs_compare/count{ii}_compare.jpg", ret)
                ii += 1
            else:
                count += 1
                cv2.waitKey(25)

    # cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # imgRW("test", "images/apple.jpg")
    # imgPyramid(cv2.imread("images/apple.jpg"))
    # imgEdge(cv2.imread("images/apple.jpg"))
    # ThreeS(cv2.imread('images/apple.jpg'))
    # imgSobel(cv2.imread('images/apple.jpg'))
    # imgScharr(cv2.imread('images/apple.jpg'))
    # imgLaplacian(cv2.imread('images/apple.jpg'))
    # imgCanny(cv2.imread('images/erode_test.jpg'))
    # imgGradient(cv2.imread('images/erode_test.jpg'))
    # imgOpen(cv2.imread('images/erode_test.jpg'))
    # imgClose(cv2.imread('images/erode_test.jpg'))
    # imgDilate(cv2.imread('images/erode_test.jpg'))
    # imgErode(cv2.imread('images/erode_test.jpg'))
    # imgFilter(cv2.imread('images/apple.jpg'))
    # imgThreshold(cv2.imread('images/apple.jpg'))
    # imgFusion(cv2.imread('images/apple.jpg'), cv2.imread('images/pear.jpg'))
    # imgMath(cv2.imread('images/apple.jpg'))
    # fillBoundary(cv2.imread('images/apple.jpg'))
    videoRW("test", "videos/test.mp4")
    # imgAttribute('images/apple.jpg')
