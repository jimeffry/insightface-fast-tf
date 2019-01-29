#!/bin/bash
#generate image list
#python image_preprocess.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_3936 --out-file ./output/bg.txt --base-label 0 --cmd-type gen_filepath_2dir
python image_preprocess.py --img-dir /home/lxy/Downloads/DataSet/insightface/Ms-1M-Celeb/train --out-file ./output/ms1m.txt  --cmd-type gen_filepath_2dir
### merge
#python image_preprocess.py  --file-in ./output/phone_moni_rc.txt  --file2-in ./output/bg_2.txt --out-file ./output/data4.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/data3.txt  --file2-in ./output/data4.txt --out-file ./output/data3_4.txt --cmd-type merge