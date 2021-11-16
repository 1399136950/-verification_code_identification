from sklearn.neighbors import KNeighborsClassifier      #利用邻近点方式训练数据
import cv2
import os
import joblib
import uuid


from handle_img import img_split, filter_noise, threshold_img


def get_feature(img):   # 提取图片特征值, 该img是已经阈值化的图像
    '''
    将整个图片的每一个像素都当做特征的话会导致数据量很大，一个100*40的图片就有4000个特征
    这里我们采用每一行的有效像素量、每一列的有效像素量和总的有效像素数量当做特征值，那么对于一个100*40的图片,总特征数量为100+40+1 = 141，可以大大缩短计算量
    '''
    line, column = img.shape
    feature_list = []   # 存放特征的列表   
    pix_count = 0
    for i in range(line):   # 每行的像素点总数当做特征值
        pix_line_count = 0
        for pix in img[i]:
            if pix == 0:
                pix_line_count += 1
            feature_list.append(pix_line_count)
        pix_count += pix_line_count
    img1 = img.T  # 图像矩阵翻转
    for i in range(column): # 每列的像素点总数当做特征值
        pix_column_count = 0
        for pix in img1[i]:
            if pix == 0:
                pix_column_count += 1
            feature_list.append(pix_column_count)
    feature_list.append(pix_count)  # 总像素点也当做一个特征
    return feature_list


def start_train(dir_name):  # 开始训练, dir_name 是 阈值后的图片的根目录,根目录下面的每个文件夹名称是图片的真实字符信息，然后每个文件夹内存放对应的图片(需要是阈值化处理过的)
    feature_list = []   # 特征集
    result_list = []    # 结果集
    for dirname in os.listdir(dir_name):
        for img_name in os.listdir(dir_name + '/' + dirname):
            img_path = dir_name + '/' + dirname + '/' + img_name
            img = cv2.imread(img_path, 0)   # 灰阶图片
            feature = get_feature(img)      # 提取特征
            feature_list.append(feature)
            result_list.append(dirname)
    knn = KNeighborsClassifier()  #引入训练方法 
    knn.fit(feature_list, result_list)  # 开始训练
    joblib.dump(knn, 'code.pkl')    # 将训练结果保存，这样下次直接从文件加载而不用重新训练


def handle_img(img_path, dest_dir_name):   # 处理图片,源图片路径 和 保存后的根目录路径
    img = cv2.imread(img_path)
    img_list = img_split(img) # 分割4块
    img_name = os.path.basename(img_path)
    realy_code = img_name.split('.')[0]  # 根据图片名称获得验证码内容
    _i = 0
    for _img in img_list:
        new_img = filter_noise(_img)    # 过滤线条干扰
        new_img_gray  = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)   # 图片灰阶
        ret, new_img_th = cv2.threshold(new_img_gray, 250, 255, cv2.THRESH_BINARY)  # 灰阶阈值化
        img_name = str(uuid.uuid1()) + '.png'   # 随机生成新的图片名称
        cv2.imwrite(dest_dir_name+'/'+realy_code[_i].lower() + '/' + img_name, new_img_th)  # 根据图片名称将图片分类保存
        _i += 1


def discriminate_code():
    knn = joblib.load('code.pkl')
    feature_list = []
    test_img_path = 'source_code/zrww.png'
    # img = cv2.imread(test_img_path)
    img_list = img_split(test_img_path) # 分割4块
    for _img in img_list:
        new_img = filter_noise(_img)    # 过滤线条干扰
        new_img_gray  = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        ret, new_img_th = cv2.threshold(new_img_gray, 250, 255, cv2.THRESH_BINARY)
        feature = get_feature(new_img_th)
        feature_list.append(feature)
    res_test = knn.predict(feature_list)
    print(res_test)


def handle_img_and_train(source_dirname, train_dirname):    # 处理图片然后训练
    '''
    source_dirname路径下的原始验证码图片必须先手动分好类，图片的名称即验证码的内容
    '''
    for img_name in os.listdir(source_dirname):
        img_path = os.path.join(source_dirname, img_name)
        handle_img(img_path, train_dirname)     # 处理图片并保存
    start_train(train_dirname)  # 开始训练


if __name__ == "__main__":
    import shutil
    
    shutil.rmtree('train_file')
    os.mkdir('train_file')
    for i in '0123456789qwertyuiopasdfghjklzxcvbnm':
        os.mkdir(f'train_file/{i}')
    handle_img_and_train('source_code', 'train_file')