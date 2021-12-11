import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 as cv
import os, shutil
from pathlib import Path
# 获得该文件夹下所有jpg图片路径
p = Path(r"F:\unsupervised learning\K-means\picture")
files = list(p.glob("**/*.png"))
# opencv读取图像 并resize为（224，224）
images = [cv.resize(cv.imread(str(file)), (224, 224)) for file in files]
paths = [file for file in files]
# 图像数组转换为float32类型并reshape  然后做归一化
images = np.array(np.float32(images).reshape(len(images), -1) / 255)
# 加载预先训练的模型MobileNetV2来实现图像分类
model = tf.keras.applications.MobileNetV2(include_top=False,
weights='imagenet', input_shape=(224, 224, 3))
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)
k = 2   # 2个类别
# K-Means聚类
kmodel = KMeans(n_clusters=k, random_state=888)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
print(kpredictions)  
a = kpredictions
a = a.tolist()
a = ','.join(str(n) for n in a)
b = open("1.txt","w")
b.write(a)
b.close()

# 预测的类别
# 0：dog    1：cat
for i in ["cat", "dog"]:
    os.mkdir(r"F:\unsupervised learning\K-means\picture_" + str(i))

# 复制文件，保留元数据 shutil.copy2('来源文件', '目标地址')
for i in range(len(paths)):
    if kpredictions[i] == 0:   
        shutil.copy2(paths[i], r"F:\unsupervised learning\K-means\picture_dog")
    else:
        shutil.copy2(paths[i], r"F:\unsupervised learning\K-means\picture_cat")