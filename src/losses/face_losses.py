import tensorflow as tf
import tensorflow.contrib.layers as tfc
import math
import numpy as np

def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(512, out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


def combine_loss(embedding, labels, class_num, margin_a, margin_m, margin_b, s):
    '''
    param embedding: the input embedding vectors
    param labels:  the input labels, the shape should be eg: (batch_size, 1)
    param s: scalar value default is 64
    param out_num: output class num
    param m: the margin value, default is 0.5
    return: the final cacualted output, this output is send into the tf.nn.softmax directly
    ls = s* ( cos(marg_a*t+m)+marg_b)
    '''
    w_init = tfc.xavier_initializer(uniform=False,seed=None,dtype=tf.float32)
    weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], class_num),
                              initializer=w_init, dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
    cos_t = tf.matmul(embedding_unit, weights_unit)
    batch_size = embedding.get_shape().as_list()[0]
    ordinal = tf.constant(list(range(0, batch_size)), tf.int32)
    labels = tf.reshape(labels,[-1])
    ordinal_y = tf.stack([ordinal, labels], axis=1)
    zy = cos_t * s
    sel_cos_t = tf.gather_nd(zy, ordinal_y)
    if margin_m != 0.0:
        cos_value = sel_cos_t / s
        thea = tf.acos(cos_value)
        thea_ax = thea * margin_a
        thea_ax_m = thea_ax + margin_m
        #condition calculate
        threshold_m = tf.cos(margin_m)
        thres_cos = tf.cos(thea_ax) + threshold_m
        #get the value margin_a* t+margin_m > pi
        zy_keep = tf.sin(thea_ax_m) - 1
        #get out
        new_zy = tf.cos(thea_ax_m)
        bg = tf.zeros_like(thres_cos)
        body = tf.where(tf.greater(thres_cos,bg),new_zy,zy_keep)
        body = body - margin_b
        new_zy = body * s
    elif margin_a == 1.0 and margin_m == 0.0:
        s_m = s * margin_b
        new_zy = sel_cos_t - s_m
    updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), zy.get_shape()))
    '''
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
    predict_cls = tf.argmax(updated_logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    predict_cls_s = tf.argmax(zy, 1)
    accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls_s, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    '''
    return updated_logits

if __name__ == '__main__':
    emb = tf.constant([[2,2,2],[1,3,3],[-1,-4,4]],dtype=tf.float32)
    w = tf.constant([[1,2],[2,2],[3,2]],dtype=tf.float32)
    label = np.array([0,1,1])
    print(emb.get_shape().as_list()[-1],label.shape)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    #err = focal_loss(pred,label,2)
    err = combine_loss(emb,label,w,2,1,1.57,0,10) #0.785
    sess.run(init)
    er_o = sess.run(err)
    print('out',er_o[0])
    print('out3',er_o[1])