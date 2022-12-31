from PIL import Image
import numpy as np

# image = Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori1.jpg") # 用PIL中的Image.open打开图像
# image_arr = np.array(image) # 转化成numpy数组
image_arr1 = np.array(Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori1.jpg"))
image_arr2 = np.array(Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori2.jpg"))
image_arr3 = np.array(Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori3.jpg"))
image_label1 = np.array(Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\aft1.png"))
image_label2 = np.array(Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\aft1.png"))
image_label3 = np.array(Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\aft1.png"))
# image_arr = [Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori1.jpg"),Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori2.jpg"),Image.open("C:\\Users\\lenovo\\Desktop\\ANN\\ori3.jpg")]
print(image_arr1)
# print(image_arr1)

w = []
w.append(image_arr1)
w.append(image_arr2)
w.append(image_arr3)

print(w)

