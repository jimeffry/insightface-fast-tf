# Reproduction of MobileNetV2 using tensorflow for insightface fast training

***
## Project Descriptions
+ **created by :** lxy 
+ **Time:**  2019/1/22 15:09
+ **project** Face Recognition
+ **company:** 
+ **rversion:** 0.1
+ **tools:**   python 2.7
+ **modified:**
+ **description:** The codes for training and testing
***
## Requests
* tensorflow >= 1.5.0
* python >= 2.7.15
* opencv >= 3.4.0
* imgaug
***
## Training Data
* The training datas are downloaded from [deepinsight](https://github.com/deepinsight/insightface)
## Run Train and Test demo
Configuration parameters lies in Root/src/configs/config.py
1. directory
+  **data** is used to store training and testing data.
+  **log** is used to store traing logs.
+  **models** is used to store network parameters.
+  **src** is used to store training and testing codes.
2. train
+  **get image list** : running Root/src/prepare_data/run_script.sh to generate traing and testing data list.
+  **image augmentation**: running Root/src/utils/transform.py for image augmentation, applying for images and images with boxes and images with keypoints
+  **pack training images**: running Root/src/prepare_data/run.sh to pack training data.
+  **to train on packed images**: running Root/src/train/run.sh
3. test
+  **test one image**: python Root/src/test/demo.py --img-path1 test.jpg --gpu 0 --load-epoch 10 --cmd-type imgtest
+  **test a video**: python Root/src/test/demo.py --file-in test.mp4 --gpu 0 --load-epoch 10 --cmd-type videotest
+  **test on a test dataset**: python demo.py --file-in ../prepare_data/output/test.txt --out-file ./output/record.txt --base-dir .../test_imgs/ --load-epoch 25 --cmd-type filetest
4. video demo for face anti-spoofing
* run Root/src/face_test/run.sh
***
## HS Demo and Properties
## Results on Test data
## Cite
* deepinsight: [deepinsight](https://github.com/deepinsight/insightface)
* insightface-TF: [infightface-tf][https://github.com/auroua/InsightFace_TF]