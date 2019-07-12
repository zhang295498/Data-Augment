# coding=utf-8

import cv2 as cv
import random
import numpy as np
from matplotlib import pyplot as plt


class DataAugment():
    """
    定义图像增广的类
    """

    #定义类初始变量
    def __init__(self,image):
        self.img = image

    #显示彩色图像
    def show_img(self):
        img = cv.imread(self.img)
        cv.imshow('img', img)
        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()

    #显示灰度图像
    def show_gray_img(self):
        img_gray = cv.imread(self.img,0)
        cv.imshow('gray_img', img_gray)
        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()

    #分离图像至B，G，R通道
    def split_img(self):

        img = cv.imread(self.img)
        B,G,R = cv.split(img)

        cv.imshow('B', B)
        cv.imshow('G', G)
        cv.imshow('R', R)

        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()

    #随机增加彩色图像各通道的灰度值
    def random_light_color(self):

        img = cv.imread(self.img)
        B,G,R = cv.split(img)

        b_randn = random.randint(-50,50)
        if b_randn == 0:
            pass
        elif b_randn > 0:
            lim = 255 - b_randn
            B[B > lim] = 255
            B[B <= lim] = (b_randn + B[B <= lim]).astype(img.dtype)
        elif b_randn < 0:
            lim = 0 - b_randn
            B[B < lim] = 0
            B[B >= lim] = (b_randn + B[B >= lim]).astype(img.dtype)

        g_randn = random.randint(-50, 50)
        if g_randn == 0:
            pass

        elif g_randn > 0:
            lim = 255 - g_randn
            G[G > lim] = 255   #数组G内的所有元素，若大于lim，则赋值为255
            G[G <= lim] = (g_randn + G[G <= lim]).astype(img.dtype)

        elif g_randn < 0:
            lim = 0 - g_randn
            G[G < lim] = 0
            G[G >= lim] = (g_randn + G[G >= lim]).astype(img.dtype)

        r_randn = random.randint(-50, 50)
        if r_randn == 0:
            pass

        elif r_randn > 0:
            lim = 255 - r_randn
            R[R > lim] = 255
            R[R <= lim] = (r_randn + R[R <= lim]).astype(img.dtype)

        elif r_randn < 0:
            lim = 0 - r_randn
            R[R < lim] = 0
            R[R >= lim] = (r_randn + R[R >= lim]).astype(img.dtype)

        print(b_randn,g_randn,r_randn)

        img_merge = cv.merge((B,G,R))

        cv.imshow('ori_img',img)

        # cv.imshow('B', B)
        # cv.imshow('G', G)
        # cv.imshow('R', R)

        cv.imshow('random_light',img_merge)
        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()

    #gamma correction
    def gamma_correcton(self,gamma=1.0):

        img = cv.imread(self.img)
        invGamma = 1.0/gamma

        table = []
        for i in range(256):
            table.append(((i/255.0) ** invGamma) * 255)

        table = np.array(table).astype("uint8") # 建立的索引表
        img_gamma = cv.LUT(img,table) #依照索引表和原始像素查找新像素

        cv.imshow('ori_img',img)
        cv.imshow('gamma_img',img_gamma)

        cv.waitKey(0)
        cv.destroyAllWindows()
    def img_his(self):

        img = cv.imread(self.img,0)
        img_blance = cv.equalizeHist(img)

        #显示图像直方图
        # plt.hist(img.flatten(), 128, [0, 256], color='purple')
        # plt.hist(img_blance.flatten(), 128, [0, 256], color='b')
        # plt.show()

        cv.imshow('ori_img',img)
        cv.imshow('blance_img',img_blance)
        cv.waitKey(0)
        cv.destroyAllWindows()
    def img_rotation(self,angle):

        img = cv.imread(self.img)

        M = cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)  # center, angle, scale
        img_rotate = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv.imshow('rotated lenna', img_rotate)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

        print(M)

        M = cv.getRotationMatrix2D((img.shape[1], img.shape[0]), angle, 1)  # center, angle, scale
        img_rotate = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv.imshow('rotated lenna', img_rotate)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

        print(M)

    def img_affine(self):
        """
        实现图像的仿射变换
        :return:
        """
        img = cv.imread(self.img)
        rows, cols, ch = img.shape
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

        M = cv.getAffineTransform(pts1, pts2)
        dst = cv.warpAffine(img, M, (cols, rows))

        cv.imshow('affine lenna', dst)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

    def per_transform(self):
        """
        perspective transform
        :return:
        """
        img = cv.imread(self.img)
        height, width, channels = img.shape

        # warp:
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv.getPerspectiveTransform(pts1, pts2)
        img_warp = cv.warpPerspective(img, M_warp, (width, height))

        cv.imshow('lenna_warp', img_warp)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

img_path = r'E:\test_img\jpeg\original\img6.jpg'

DA = DataAugment(img_path)

DA.show_img()
DA.show_gray_img()
DA.gamma_correcton(2.0)
DA.random_light_color()
DA.img_affine()
DA.img_rotation(30)
DA.split_img()
