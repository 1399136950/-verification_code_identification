import cv2
import uuid
import os
import collections


def match_img(src_path, temp_path='temp.jpg'):  
    """
    模板匹配
    """
    srcImage = cv2.imread(src_path, 1) # 读取图片
    templateImage = cv2.imread(temp_path, 1) # 读取模板图片
    h, w = templateImage.shape[:2]  # 获取模板的高和宽
    res = cv2.matchTemplate(srcImage, templateImage, cv2.TM_CCOEFF_NORMED) # 进行匹配
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # 使用cv2.minMaxLoc()函数可以得到最大匹配值的坐标
    left_top = max_loc  # 左上角
    return max_val, left_top


def filter_noise(img):
    """
    图片去噪
    """
    dicts = collections.defaultdict(int)  # 用于记录每一个像素的数量
    pixel_type_coordinate = collections.defaultdict(list)  # 记录每种像素类型的坐标
    l, c, _ = img.shape
    for i in range(l):
        for j in range(c):
            pixel = img[i][j].tobytes()
            dicts[pixel] += 1
            # 记录每一个像素类型的具体坐标
            pixel_type_coordinate[pixel].append([i, j])
    sorted_list = sorted(dicts, key=lambda x:dicts[x], reverse=True)    # 根据像素数量的多少排序，返回一个像素类型列表,像素最多的在最前
    
    # 在每一个图片中，数量最多的像素肯定是空白区域，其次是组成字符的像素，剩余的其他所有像素可以视为噪点消除掉
    # 去掉数量最多的两个像素类型
    dicts.pop(sorted_list[0])
    dicts.pop(sorted_list[1])
    
    # 剩下的像素类型都视为噪点
    for pixel_type in dicts:
        pisel_list = pixel_type_coordinate[pixel_type]  # 获取该像素类型的所有坐标
        for x, y in pisel_list: # 遍历坐标，将其像素改为[255,255,255](去噪)
            img[x][y] = [255, 255, 255]   
    return img  # 返回去噪后的图片


def img_split(img, width=25):
    """
    图片切片
    """
    l,c,_= img.shape
    start = 0
    img_list = []
    for i in range(4):
        # 分析得出一个验证码宽100pxl, 因此单个字符占25pxl
        img1 = img[:, start:start+width, :]
        start += width
        img_list.append(img1)
    return img_list


def img_split_classify(img_path, width=25):   # 图片切片和分类
    realy_code = os.path.basename(img_path).split('.')[0]
    print(img_path)
    img = cv2.imread(img_path)
    l,c,_= img.shape
    start = 0
    for i in range(4):
        img1 = img[:,start:start+width,:]
        new_img = filter_noise(img1)
        start += width
        img_name = str(uuid.uuid1())+'.png'
        cv2.imwrite('class/'+realy_code[i].lower() + '/' + img_name, new_img)


def threshold_img(src_path, dst_path=None):  # 阈值化图片并保存
    img_gray = cv2.imread(src_path, 0)
    ret, img_th = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
    cv2.imwrite(dst_path, img_th)


if __name__ == '__main__':
    img = cv2.imread('2.png')
    new_img = img[:,1:-1,:]
    img_list = img_split(new_img, 32)
    for i in img_list:
        cv2.imshow('', i)
        cv2.waitKey(0)
        cv2.imshow('', filter_noise(i))
        cv2.waitKey(0)
