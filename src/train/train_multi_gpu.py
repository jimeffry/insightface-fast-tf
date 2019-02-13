# -*- coding:utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/10 15:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib as tfc
import os, sys
import numpy as np
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
import mobilenetV2
import resnet
import lenet5
sys.path.append(os.path.join(os.path.dirname(__file__),"../prepare_data"))
from read_tfrecord import Read_Tfrecord
from read_multi_tfrecord import read_multi_rd
from convert_data_to_tfrecord import label_show
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from get_property import load_property
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from loss import focal_loss,cal_accuracy,entropy_loss
from face_losses import combine_loss

def parms():
    parser = argparse.ArgumentParser(description='Face-anti training')
    parser.add_argument('--load-num',dest='load_num',type=str,default=None,help='ckpt num')
    parser.add_argument('--save-weight-period',dest='save_weight_period',type=int,default=5,\
                        help='the period to save')
    parser.add_argument('--epochs',type=int,default=20000,help='train epoch nums')
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=32,\
                        help='train batch size')
    parser.add_argument('--model-dir',dest='model_dir',type=str,default='../../models/',\
                        help='path saved models')
    parser.add_argument('--log-path',dest='log_path',type=str,default='../../logs',\
                        help='path saved logs')
    parser.add_argument('--gpu-list',dest='gpu_list',type=str,default='0',\
                        help='train on gpu num')
    parser.add_argument('--tower-name',dest='tower_name',type=str,default='tower',\
                        help='train on gpu num')
    parser.add_argument('--property-file',dest='property_file',type=str,\
                        default='../../data/property.txt',help='nums of train dataset images')
    parser.add_argument('--data-record-dir',dest='data_record_dir',type=str,\
                        default='../../data/',help='tensorflow data record')
    parser.add_argument('--margin-a',dest='margin_a',type=float,default=1,help='cos(ax+m)+b')
    parser.add_argument('--margin-b',dest='margin_b',type=float,default=0.0,help='cos(ax+m)+b')
    parser.add_argument('--margin-m',dest='margin_m',type=float,default=0.0,help='cos(ax+m)+b')
    parser.add_argument('--scale-s',dest='scale_s',type=int,default=1,help='s*[cos(ax+m)+b]')
    return parser.parse_args()

def get_model(img_batch,class_nums):
    if cfgs.NET_NAME in ['mobilenetv2']:
        logits = mobilenetV2.get_symble(img_batch,w_decay=cfgs.WEIGHT_DECAY,\
                                       class_num=class_nums,train_fg=True)
    elif cfgs.NET_NAME in ['resnet50','resnet100']:
        logits = resnet.get_symble(img_batch,w_decay=cfgs.WEIGHT_DECAY,\
                                    class_num=class_nums,train_fg=True)
    elif cfgs.NET_NAME in ['lenet5','lenet']:
        logits = lenet5.get_symble(img_batch,w_decay=cfgs.WEIGHT_DECAY,\
                                       class_num=class_nums,train_fg=True)
        #logits = lenet5.get_symble(img_batch,class_num=class_nums)
    return logits

def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_gradients(tower_gradients):
    average_grads = []
    # Run this on cpu_device to conserve GPU memory
    with tf.device('/cpu:0'):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []
            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])
            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)
    # Return result to caller
    return average_grads

def train(args):
    model_dir = args.model_dir
    model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir,cfgs.MODEL_PREFIX)
    load_num = args.load_num
    log_dir = args.log_path
    epochs = args.epochs
    batch_size = args.batch_size
    save_weight_period = args.save_weight_period
    data_record_dir = args.data_record_dir
    margin_a = args.margin_a
    margin_b = args.margin_b
    margin_m = args.margin_m
    scale_s = args.scale_s
    gpu_list = [int(i) for i in args.gpu_list.split(',')]
    print("train gpu:",gpu_list)
    gpu_num = len(gpu_list)
    property_file = os.path.join(data_record_dir,cfgs.DATASET_NAME,'property.txt')
    Property = load_property(property_file)
    train_img_nums = Property['img_nums']
    class_nums = Property['cls_num']
    # ----------------------------------------------------------------------------------------------------build graph
    with tf.Graph().as_default():
        # ----------------------------------------------------------------------------------------------------get rd data
        with tf.variable_scope('get_batch'):
            if cfgs.RD_MULT:
                img_batch,label_batch = read_multi_rd(data_record_dir,'fg','bg',batch_size,train_img_nums,1.0/class_nums)
            else:
                tfrd = Read_Tfrecord(cfgs.DATASET_NAME,data_record_dir,batch_size,20000)#train_img_nums)
                img_batch, label_batch = tfrd.next_batch()
        #----------------------------------------------------------------------------------------------------build placeholder
        #images_input = tf.placeholder(name='img_inputs', shape=[None,cfgs.IMG_SIZE[0],cfgs.IMG_SIZE[1],3], dtype=tf.float32)
        #labels_input = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)
        # splits input to different gpu
        images_s = tf.split(img_batch, num_or_size_splits=gpu_num, axis=0)
        labels_s = tf.split(label_batch, num_or_size_splits=gpu_num, axis=0)
        # -----------------------------------------------------------------------------------------learning rate and optimizer
        global_step = tf.train.create_global_step()
        lr = tf.train.piecewise_constant(global_step,
                                        boundaries=[np.int64(x) for x in cfgs.DECAY_STEP],
                                        values=[y for y in cfgs.LR])
        tf.summary.scalar('lr', lr)
        optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
        #optimizer = tf.train.AdamOptimizer(lr)
        # -----------------------------------------------------------------------------------get multi-model net and build loss
        # Calculate the gradients for each model tower.
        tower_grads = []
        loss_dict = {}
        loss_keys = []
        for i,idx in enumerate(gpu_list):
            with tf.variable_scope(tf.get_variable_scope(),reuse= i > 0):
                with tf.device('/gpu:%d' % idx):
                    with tf.name_scope('%s_%d' % (args.tower_name, idx)) as scope:
                        features = get_model(images_s[i],cfgs.FEATURE_LEN)
                        logits = combine_loss(features,labels_s[i],class_nums,margin_a,margin_m,margin_b,scale_s)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # define the cross entropy
                        cls_loss,soft_logits = entropy_loss(logits,labels_s[i],class_nums)
                        # define weight deacy losses
                        wd_loss = tf.add_n(tf.losses.get_regularization_losses())
                        total_loss = cls_loss + wd_loss
                        loss_dict[('cls_loss_%s_%d' % ('gpu', i))] = cls_loss
                        loss_keys.append(('cls_loss_%s_%d' % ('gpu', i)))
                        loss_dict[('wd_loss_%s_%d' % ('gpu', i))] = wd_loss
                        loss_keys.append(('wd_loss_%s_%d' % ('gpu', i)))
                        loss_dict[('total_loss_%s_%d' % ('gpu', i))] = total_loss
                        loss_keys.append(('total_loss_%s_%d' % ('gpu', i)))
                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)
                        tf.add_to_collection("total_loss",total_loss)
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                            acc_op,label_out,pred = cal_accuracy(soft_logits,labels_s[i])
        # ---------------------------------------------------------------------------------------------compute gradients
        grads = average_gradients(tower_grads)
        with tf.control_dependencies(update_ops):
            # Apply the gradients to adjust the shared variables.
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_loss = tf.reduce_mean(tf.get_collection("total_loss"),0)
        #train_op = optimizer.minimize(train_loss,colocate_gradients_with_ops=True)
        # ---------------------------------------------------------------------------------------------------add summary
        tf.summary.scalar('LOSS/cls_loss', cls_loss)
        tf.summary.scalar('LOSS/total_loss', total_loss)
        tf.summary.scalar('LOSS/regular_weights', wd_loss)
        tf.summary.scalar('LOSS/train_loss',train_loss)
        summary_op = tf.summary.merge_all()
        # ---------------------------------------------------------------------------------------------------train steps
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=30)
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        with tf.Session(config=tf_config) as sess:
            sess.run(init_op)
            if load_num is not None:
                model_path = "%s-%s" %(model_path,str(load_num))
                model_dict = '/'.join(model_path.split('/')[:-1])
                ckpt = tf.train.get_checkpoint_state(model_dict)
                print("restore model path:",model_path)
                readstate = ckpt and ckpt.model_checkpoint_path
                assert readstate, "the params dictionary is not valid"
                saver.restore(sess, model_path)
                print("restore models' param")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            summary_path = os.path.join(log_dir,'summary')
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
            #img_dict = dict()
            try:
                with tf.Graph().as_default():
                    for epoch_tmp in range(epochs):
                        for step in range(np.ceil(train_img_nums/batch_size).astype(np.int32)):
                            training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                            #tmp_img,tmp_label = sess.run([img_batch,label_batch])
                            #_,_ = sess.run([train_op,global_step]) 
                            if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                                _, global_stepnp = sess.run([train_op, global_step])
                                #pass
                            else:
                                if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                                    start = time.time()
                                    label_o,pred_out,global_stepnp, totalLoss,cls_l,acc,cur_lr = sess.run([label_out,pred,global_step, train_loss,cls_loss,acc_op,lr])
                                    end = time.time()
                                    print('label',label_o)
                                    print('pred',pred_out)
                                    #print('feature',feat[0])
                                    print(""" %s epoch:%d step:%d | per_cost_time:%.3f s | total_loss:%.3f | cls_loss:%.5f | acc:%.4f | lr:%.6f""" \
                                        % (str(training_time), epoch_tmp,global_stepnp, (end - start),totalLoss,cls_l,acc,cur_lr))
                                else:
                                    if step % cfgs.SMRY_ITER == 0:
                                        global_stepnp, summary_str = sess.run([global_step, summary_op])
                                        summary_writer.add_summary(summary_str, global_stepnp)
                                        summary_writer.flush()
                        if (epoch_tmp > 0 and epoch_tmp % save_weight_period == 0) or (epoch_tmp == epochs - 1):
                            save_dir = model_path
                            saver.save(sess, save_dir,epoch_tmp)
                            print(' weights had been saved')
            except tf.errors.OutOfRangeError:
                print("Trianing is over")
            finally:
                coord.request_stop()
                summary_writer.close()
            coord.join(threads)
            #record_file_out.close()
            sess.close()

if __name__ == '__main__':
    args = parms()
    gpu_group = args.gpu_list
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    train(args)




