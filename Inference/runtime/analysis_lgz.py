import cv2
import os
import re
import numpy as np
import statistics

def cal_contour(mask):
    h, w = mask.shape

    counter_list = []

                    
    hrz_list = [sum(row == 255) for row in mask]
    hrz_nz_list = [i for i in hrz_list if i != 0]
    counter_list.extend(hrz_nz_list)
    
                    
    vrt_list = [sum(mask[:, j] == 255) for j in range(w)]
    vrt_nz_list = [i for i in vrt_list if i != 0]
    counter_list.extend(vrt_nz_list)

                                         
    length = statistics.mode(counter_list)                                                                                                                               
    length = 192

                                                                
    hrz_list = [h * w if i == length else max(h, w) - abs(i - length) for i in hrz_list]
    vrt_list = [h * w if i == length else max(h, w) - abs(i - length) for i in vrt_list]
    
                       
    upbound = 0
    tempy = 0

    for i in range(h - length):
        temp = sum(hrz_list[i:i + length])
        if  temp > upbound:
            upbound = temp
            tempy = i
    
                       
    upbound = 0
    tempx = 0

    for j in range(w - length):
        temp = sum(vrt_list[j:j + length])
        if temp > upbound:
            upbound = temp
            tempx = j


                                    
                                    
                         
    coordinates = [[tempx, tempy], [tempx + length - 1, tempy], [tempx + length - 1, tempy + length - 1], [tempx, tempy + length - 1]]                                                                     
                             
                        
    return coordinates


if __name__ == '__main__':

                
    src_folder = "/new_groupsahre_2/screen_watermark/screen_watermark_results_shibai"
    dest_folder = "/new_groupsahre_2/screen_watermark/screen_watermark_results_failed_1"

               
    os.makedirs(dest_folder, exist_ok=True)
    print(f"目标文件夹 {dest_folder} 已创建或已存在。")

                   
    pattern = re.compile(r'\{(\d+)\}_\{(\d\.\d+)\}.png')

             
    corner_coordinates = {}
                 
    for filename in os.listdir(src_folder):
        if filename.endswith(".png"):
            match = pattern.match(filename)
            if match:
                image_number, accuracy = match.groups()
                accuracy = float(accuracy)

                                 
                if accuracy < 1.0:
                    print(f"处理图像 {filename}，准确率为 {accuracy}")

                          
                    image_path = os.path.join(src_folder, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                              
                    cv2.imwrite(os.path.join(dest_folder, filename), image)
                    print(f"图像 {filename} 已复制到 {dest_folder}")
                    '''
                    # 使用cv2.findContours寻找轮廓
                    # 第一个参数是源图像，这里是二值图像
                    # cv2.RETR_EXTERNAL 表示只检测外轮廓
                    # cv2.CHAIN_APPROX_SIMPLE 是轮廓近似方法，它只保留轮廓的拐点信息，压缩垂直、水平和对角线段，只保留它们的终点
                    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 从检测到的所有轮廓中找出最大的一个
                    # max函数用于找出最大元素，key=cv2.contourArea 表示比较标准是轮廓的面积
                    # 这假设最大的轮廓是我们关注的对象，即白色正方形
                    largest_contour = max(contours, key=cv2.contourArea)

                    # 寻找轮廓的四个角
                    # cv2.minAreaRect函数计算轮廓的最小外接矩形
                    # 它返回一个Box2D结构 - 包含中心点坐标、宽高和旋转角度
                    rect = cv2.minAreaRect(largest_contour)

                    # cv2.boxPoints(rect) 从Box2D结构中获得矩形的四个角点
                    box = cv2.boxPoints(rect)

                    # np.int0(box) 转换坐标为整数
                    box = np.int0(box)

                    # 保存坐标
                    # 将四个角点坐标转换为列表并保存到字典中
                    # 字典的键是图像的序号，值是轮廓的四个角点坐标
                    '''
                                                                    
                    corner_coordinates[image_number] = cal_contour(image)
                    print(f"图像 {filename} 的拐角坐标已保存。")

                     
    coordinates_file = os.path.join(dest_folder, 'corner_coordinates.txt')
    with open(coordinates_file, 'w') as f:
        for image_number in sorted(corner_coordinates, key=lambda x: int(x)):
            corners = corner_coordinates[image_number]
            f.write(f"{image_number}: {corners}\n")

    print(f"所有拐角坐标已按图像序号排序并保存到 {coordinates_file}")
