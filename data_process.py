import cv2

import numpy as np
import os
import glob
# 设置窗口大小（宽度 x 高度）
window_width = 800
window_height = 800

# 设置图片的位置在窗口的左上角（x, y 坐标）
window_x = 50
window_y = 50

def count_cell(image,ori_image):#原图、预测图、保存路径
    # 读取灰度图像
    #image_path = r'C:\Users\lenovo\Desktop\unet\label3\28.png'

    gray = image
    cv2.imwrite('104.png',gray)

    gray1 = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波(低通,去掉高频分量,图像平滑)

    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)

    # 上面处理了多次,为消除细胞体内的浓淡深浅差别,避免在后续运算中产生孔洞


    gray2 = gray1  # 0~255反相,为了适应腐蚀算法
    '''
    cv2.imshow("draw11", gray2)
    cv2.moveWindow("draw11", window_x, window_y)
    '''
    # ret, thresh = cv2.threshold(gray2, 80,255, cv2.THRESH_BINARY) # 固定 阈值处理 二值化图像
    '''
    # 进行手动阈值处理并将图像二值化
    threshold_value=200
    ret, thresh = cv2.threshold(gray2, threshold_value, 255, cv2.THRESH_BINARY)
    '''
    '''
    # 设置特定像素值为某一值，其他像素设为黑色
    specific_pixel_value = 3  # 用于指定要替换的像素值
    replace_value = 255         # 将特定像素值替换为此值
    gray2[gray2 == specific_pixel_value] = replace_value
    thresh=gray2
    '''
    '''
    cv2.imshow("draw22", thresh)
    cv2.moveWindow("draw22", window_x, window_y)
    '''
    # # 下面为了去除细胞之间的粘连,以免把两个细胞计算成一个
    #
    # kernel = np.ones((2, 2), np.uint8)  # 进行腐蚀操作,kernel=初值为1的2*2数组
    #
    # erosion = cv2.erode(thresh, kernel, iterations=10)  # 腐蚀:卷积核沿着图像滑动，如果与卷积核对应的原图像的所有像素值都是1，那么中心元素就保持原来的像素值，否则就变为零。

    # 定义核（结构元素），用于腐蚀和膨胀操作
    kernel = np.ones((3, 3), np.uint8)  # 可根据需要调整核的大小

    # 开运算操作（先腐蚀后膨胀）
    erosion = cv2.morphologyEx(gray2, cv2.MORPH_OPEN, kernel)
    '''
    cv2.imshow("draw33", erosion)
    cv2.moveWindow("draw33", window_x, window_y)
    '''
    '''
    #我们不用回复size
    # 下面为恢复一点SIZE,因为上面多次腐蚀,块太小了,所以这里膨胀几次
    
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    
    cv2.imshow("draw44", dilation)
    cv2.moveWindow("draw44", window_x, window_y)
    '''

    '''
    dilation=erosion
    contours, hirearchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 上面查找出块轮廓(实现细胞计数)

    # 对连通域面积进行比较

    contours_out = []  # 建立空list，放减去最小面积的数

    for i in contours:

        if (cv2.contourArea(i) > 2 and cv2.contourArea(i) < 3000 and i.shape[0] > 2):
            # 排除面积太小或太大的块轮廓 而且排除 "点数不足2"的离散点块

            contours_out.append(i)

    total_num = len(contours_out)

    print("Count=%d" % total_num)  # 输出计算细包个数

    # 下面生成彩色结构的灰度图像,为的是在上面画彩色标注

    color_of_gray_img = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(color_of_gray_img, contours_out, -1, (50, 50, 250), 2)  # 用红色线,描绘块轮廓

    # 求连通域重心 以及 在重心坐标点描绘数字

    for i, j in zip(contours_out, range(total_num)):
        M = cv2.moments(i)

        cX = int(M["m10"] / M["m00"])

        cY = int(M["m01"] / M["m00"])

        cv2.putText(color_of_gray_img, str(j + 1), (cX + 10, cY), 1, 1, (50, 250, 50), 1)  # 在中心坐标点上描绘数字

    strout = "Count=%d" % total_num  # 输出计算细包个数

    print(strout)

    cv2.putText(color_of_gray_img, strout, (2, 12), 1, 1, (250, 150, 150), 1)
    '''
    binary_image=erosion
    # 查找所有轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 定义圆形度阈值，可以根据需要调整
    circularity_threshold = 0.2

    # 定义颜色映射表
    color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # 记录每个连通块的信息
    results = []

    # 将二值图像转换为彩色图像
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # 定义自定义颜色映射表，按照连通块编号指定颜色，如编号1对应(255, 0, 0)，编号2对应(0, 255, 0)，以此类推
    custom_color_map = {
        1: (255, 0, 0),  # 红色
        2: (0, 255, 0),  # 绿色
        3: (0, 0, 255),  # 蓝色
        # 可根据需要继续添加其他编号对应的颜色
    }

    count = 0  # 连通块总数

    # 遍历每个轮廓，并计算圆形度、面积以及坐标信息
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 筛选近似圆形的连通块
        if circularity > circularity_threshold:
            # 记录连通块信息
            coordinates = [tuple(point[0]) for point in contour]  # 提取坐标点
            results.append({
                'index': i + 1,
                'circularity': circularity,
                'area': area,
                'coordinates': coordinates
            })

            # 对符合要求的连通块进行彩色标注编号
            if (i + 1) in custom_color_map:
                color = custom_color_map[i + 1]
            else:
                color_index = (i // len(color_map)) % len(color_map)  # 防止超出预定义颜色映射表的索引范围
                color = color_map[color_index]

            cv2.drawContours(color_image, [contour], -1, color, thickness=-1)  # thickness=-1是填充整个连通块，2就是边缘圈起来
            cv2.putText(color_image, str(i + 1), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # 计算连通块的总数
    count = len(results)
    # 获取图像的高度和宽度
    height, width, _ = color_image.shape
    font_scale = 3.0

    # 设置文本
    text = f"CELL COUNT = {count}"

    # 设置文本的颜色 (BGR格式)
    text_color = (0, 255, 0)  # 这里是绿色

    # 获取文本的大小
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

    # 在图片左上角绘制文本
    text_x = 10
    text_y = 10 + text_size[1]  # 文本的高度，用于设置绘制文本的位置
    cv2.putText(color_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

    # 保存连通块信息到 txt 文件
    result_string = ""
    result_string += f"图片菌落总数: {count} \n"
    # # 保存图片
    # cv2.imwrite('101.png', color_image)
    # cv2.waitKey()

    # 合并图片
    merge_image=merge(ori_image,color_image)
    # cv2.destroyWindow()

    return result_string, merge_image
    # 输入图像你可以直接抓图(上面那张),存成jpg或png,和上面的python代码放在同一个目录下运行.

def merge(original_image,label_image ):
    '''
    # 调整二值标签图的通道数，使其与原始图像相同
    label_image = cv2.merge((label_image, label_image, label_image))
    '''
    # 将标签图与原始图像进行重叠
    alpha = 0.5  # 设置原始图像的权重
    beta = 1.0 - alpha  # 设置二值标签图的权重
    original_image=np.array(original_image)
    #cv2.imwrite('103.png',original_image)
    overlay_image = cv2.addWeighted(original_image, alpha, label_image, beta, 0)
    '''
    # 显示原始图像和重叠后的图像
    cv2.imshow('Original Image', original_image)
    cv2.moveWindow("Original Image", window_x, window_y)
    cv2.imshow('Overlay Image', overlay_image)
    cv2.moveWindow("Overlay Image", window_x, window_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # 保存图片

    # cv2.imwrite('102.png', overlay_image)
    # cv2.waitKey()
    return overlay_image

if __name__ == "__main__":

    path=r'C:\Users\lenovo\Desktop\unet\img'#原图文件夹

    # 获取文件夹中所有图片文件的路径
    image_paths = glob.glob(os.path.join(path, '*.png'))  # 以 png 格式为例，您可以根据实际情况更改后

    # 检查文件夹是否为空
    if not image_paths:
        print("文件夹中没有图片文件。")
    else:
        # 遍历图片文件
        for image_path in image_paths:
            image_name = os.path.basename(image_path)#图片名字
            '''
            # 分离文件名和后缀
            image_name, image_extension = os.path.splitext(os.path.basename(image_path))
            '''
            #计算细胞数目--（二值图）
            print()
            binary_image_path = os.path.join(r'C:\Users\lenovo\Desktop\unet\label3', image_name)
            count_cell(binary_image_path,os.path.join(r'C:\Users\lenovo\Desktop\unet\count', image_name))

            #叠加两个图片，（二值图与rgb，或者rgb与rgb）
            original_image_path = image_path
            label_image_path = os.path.join(r'C:\Users\lenovo\Desktop\unet\count', image_name)
            merge(original_image_path,label_image_path,os.path.join(r'C:\Users\lenovo\Desktop\unet\merge', image_name))


