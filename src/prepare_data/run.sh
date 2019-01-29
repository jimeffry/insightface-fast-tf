#!/bin/bash
python convert_data_to_tfrecord.py --image-dir /home/lxy/Downloads/DataSet/insightface/Ms-1M-Celeb/train --save-name train --dataset-name MS1M \
        --anno-file ./output/ms1m.txt

###convert multi dataset
#python convert_data_to_tfrecord.py --image-dir /home/lxy/Downloads/DataSet/Face_reg/id_5000_org --save-name fg --dataset-name Prison \
 #       --anno-file ./output/fg.txt
#python convert_data_to_tfrecord.py --image-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_3936 --save-name bg --dataset-name Prison \
 #       --anno-file ./output/bg.txt
