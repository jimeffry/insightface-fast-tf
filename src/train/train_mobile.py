import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords import parse_function
import os
from nets.FaceMobileNet import MobileNetFace
from losses.face_losses import arcface_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=50,type=int, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=100000,type=int, help='epoch to train the network')
    parser.add_argument('--batch_size', default=16, type=int,help='batch size to train network')
    parser.add_argument('--lr_steps', default=[4000, 6000, 8000],nargs="+", help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9,type=float, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4,type=float, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['cfp_ff'],nargs="+", help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112',type=str, help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112],nargs="+", help='the image size')
    parser.add_argument('--num_output', default=85164,type=int, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/mobileface/summary',type=str, help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/mobileface/ckpt',type=str, help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/mobileface/logs',type=str, help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100,type=int, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, type=int,help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, type=bool,help='show device placement log')
    parser.add_argument('--summary_interval', default=300, type=int,help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10,type=int, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000,type=int, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=1000,type=int, help='intervals to save ckpt file')
    parser.add_argument('--gpu', default='0', type=str,help='which gpu to run')
    parser.add_argument('--load_epoch', default=0,type=int, help='the model file ')
    parser.add_argument('--wdecay', default=0.0005,type=float, help='conv weights decay ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 1. define global parameters
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    input_size_h,input_size_w = args.image_size
    images = tf.placeholder(name='img_inputs', shape=[None, input_size_h,input_size_w, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    dataset = tf.contrib.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    model_net = MobileNetFace(images,is_training=True,wdecay=args.wdecay)
    net_out = model_net.get_mobileface()
    # 3.2 get arcface loss
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    logit = arcface_loss(embedding=net_out, labels=labels, w_init=w_init_method, out_num=args.num_output)
    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    # 3.4 define weight deacy losses
    # for var in tf.trainable_variables():
    #     print(var.name)
    # print('##########'*30)
    # 3.5 total losses
    total_loss = tf.add_n(tf.losses.get_regularization_losses())
    # 3.6 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [p*val for val in args.lr_steps]
    print(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # 3.11 summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # 3.13 init all variables
    if args.load_epoch ==0:
        print("gloable init")
        sess.run(tf.global_variables_initializer())
    else:
        prefix = "InsightFace_iter_"
        prefix = os.path.join(args.ckpt_path,prefix)
        prefix = prefix +str(args.load_epoch)+".ckpt"
        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        readstate = ckpt and ckpt.model_checkpoint_path
        assert  readstate, "the params dictionary is not valid"
        print("model path: ",prefix)
        saver.restore(sess,prefix)
        print("saver done")
    # 4 begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    # 4 begin iteration
    count = 0
    total_accuracy = {}
    for i in range(args.epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                images_train = (images_train -127.5) * 0.0078125
                #images_train = cv2.resize(images_train,(112,112))
                feed_dict = {images: images_train, labels: labels_train}
                start = time.time()
                _, total_loss_val, inference_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, inc_op, acc],
                              feed_dict=feed_dict
                              )
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, \
                         training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_val, inference_loss_val, acc_val, pre_sec))
                count += 1
                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                # validate
                if count > 0 and count % args.validate_interval == 0:
                    feed_dict_test = dict()
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=net_out, batch_size=args.batch_size, feed_dict=feed_dict_test,
                             input_placeholder=images)
                    print('test accuracy is: ', str(results[0]))
                    total_accuracy[str(count)] = results[0]
                    log_file.write('########'*10+'\n')
                    log_file.write(','.join(list(total_accuracy.keys())) + '\n')
                    log_file.write(','.join([str(val) for val in list(total_accuracy.values())])+'\n')
                    log_file.flush()
                    if max(results) > 0.996:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
                        log_file.write('######Best Accuracy######'+'\n')
                        log_file.write(str(max(results))+'\n')
                        log_file.write(filename+'\n')
                        log_file.flush()
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
        # save ckpt files
        if i > 0 and i % args.ckpt_interval == 0:
            filename = 'InsightFace_iter_{:d}'.format(i) + '.ckpt'
            filename = os.path.join(args.ckpt_path, filename)
            saver.save(sess, filename)
    log_file.close()
    #log_file.write('\n')