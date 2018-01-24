# alexnet + weights comes from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

import tensorflow as tf
from makePatches import createPatches
from data_handling_functions import make_dataset_from_patch
import numpy as np
import os
import matplotlib.pyplot as plt
from caffe_classes import class_names

def main():

    MODEL_ID = 666
    N_HIDDEN = 512
    resize_factor = 1.0
    batch_size = 125

    n_runs = 10

    for run in range(n_runs):
        print('###########################################################################')
        print('THIS IS RUN '+str(run))
        print('###########################################################################')
        for STIM in ['vernier', 'crowded', 'uncrowded']:
            if 'vernier' in STIM:
                for TRAINING in [True, False]:
                    tf.reset_default_graph()
                    run_alexcrowd_session(MODEL_ID=MODEL_ID, VERSION=run, STIM=STIM, N_HIDDEN=N_HIDDEN,
                                          TRAINING=TRAINING, resize_factor=resize_factor, batch_size=batch_size,
                                          total_n_samples=2000*batch_size, scope=(str(run)+'_'+STIM+'_'+str(TRAINING)))
            else:
                TRAINING = False
                tf.reset_default_graph()
                run_alexcrowd_session(MODEL_ID=MODEL_ID, VERSION=run, STIM=STIM, N_HIDDEN=N_HIDDEN,
                                      TRAINING=TRAINING, resize_factor=resize_factor, batch_size=batch_size,
                                      total_n_samples=3000, scope=(str(run)+'_'+STIM+'_'+str(TRAINING)))


def run_alexcrowd_session(MODEL_ID, VERSION, STIM, N_HIDDEN, TRAINING, resize_factor, batch_size, total_n_samples, scope):

    ####################################################################################################################
    # Model name and logdir. Choose to train or not. Checkpoint for model saving
    ####################################################################################################################


    if N_HIDDEN is None:
        MODEL_NAME = 'alexcrowd_batch_norm_' + str(MODEL_ID)
        LOGDIR = MODEL_NAME + '_logdir/version_' + str(VERSION) + '_resize_' + str(resize_factor)
    else:
        MODEL_NAME = 'alexcrowd_batch_norm_' + str(MODEL_ID)
        LOGDIR = MODEL_NAME + '_logdir/version_' + str(VERSION) + '_hidden_' + str(N_HIDDEN) + '_resize_' + str(resize_factor)
    checkpoint_path = LOGDIR + '/' + MODEL_NAME + '_hidden_' + str(N_HIDDEN) + '_resize_' + str(resize_factor) + "_model.ckpt"
    restore_checkpoint = True
    continue_training_from_checkpoint = False

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)


    ####################################################################################################################
    # Data handling (we will create data in batches later, during the training/testing)
    ####################################################################################################################


    n_batches = total_n_samples//batch_size

    # save parameters
    if TRAINING is True:
        filename = LOGDIR + '/' + STIM + '_training_parameters.txt'
    else:
        filename = LOGDIR + '/' + STIM + '_testing_parameters.txt'
    with open(filename, 'w') as f:
        f.write("Parameter\tvalue\n")
        variables = locals()
        variables = {key: value for key, value in variables.items()
                     if 'set' not in key}
        f.write(repr(variables))
    print('Parameter values saved.')


    ####################################################################################################################
    # Network weights and structure
    ####################################################################################################################


    x = tf.placeholder(tf.float32, [None, 227, 227, 3], name='input_image')
    tf.summary.image('input', x, 6)
    y = tf.placeholder(tf.int64, [None], name='input_label')
    is_training = tf.placeholder(tf.bool, (), name='is_training')

    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    with tf.name_scope('conv1'):
        k_h = 11
        k_w = 11
        c_o = 96
        s_h = 4
        s_w = 4
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        # tf.summary.histogram('conv1',conv1)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier1'):
        classifier1 = vernier_classifier(conv1, is_training, N_HIDDEN, name='classifier1')
        x_entropy1 = vernier_x_entropy(classifier1,y)
        correct_mean1 = vernier_correct_mean(tf.argmax(classifier1, axis=1), y) # match correct prediction to each entry in y
        train_op1 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy1,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier1'),
                                                       name="training_op")

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    with tf.name_scope('lrn1'):
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    with tf.name_scope('maxpool1'):
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    with tf.name_scope('conv2'):
        k_h = 5
        k_w = 5
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        # tf.summary.histogram('conv2',conv2)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier2'):
        classifier2 = vernier_classifier(conv2, is_training, N_HIDDEN, name='classifier2')
        x_entropy2 = vernier_x_entropy(classifier2,y)
        correct_mean2 = vernier_correct_mean(tf.argmax(classifier2, axis=1), y) # match correct prediction to each entry in y
        train_op2 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy2,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier2'),
                                                       name="training_op")

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    with tf.name_scope('lrn2'):
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    with tf.name_scope('maxpool2'):
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    with tf.name_scope('conv3'):
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        # tf.summary.histogram('conv3',conv3)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier3'):
        classifier3 = vernier_classifier(conv3, is_training, N_HIDDEN, name='classifier3')
        x_entropy3 = vernier_x_entropy(classifier3,y)
        correct_mean3 = vernier_correct_mean(tf.argmax(classifier3, axis=1), y)  # match correct prediction to each entry in y
        train_op3 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy3,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier3'),
                                                       name="training_op")

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    with tf.name_scope('conv4'):
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        # tf.summary.histogram('conv4',conv4)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier4'):
        classifier4 = vernier_classifier(conv4, is_training, N_HIDDEN, name='classifier4')
        x_entropy4 = vernier_x_entropy(classifier4,y)
        correct_mean4 = vernier_correct_mean(tf.argmax(classifier4, axis=1), y)  # match correct prediction to each entry in y
        train_op4 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy4,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier4'),
                                                       name="training_op")

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    with tf.name_scope('conv5'):
        k_h = 3
        k_w = 3
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        # tf.summary.histogram('conv5', conv5)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier5'):
        classifier5 = vernier_classifier(conv5, is_training, N_HIDDEN, name='classifier5')
        x_entropy5 = vernier_x_entropy(classifier5,y)
        correct_mean5 = vernier_correct_mean(tf.argmax(classifier5, axis=1), y)  # match correct prediction to each entry in y
        train_op5 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy5,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier5'),
                                                       name="training_op")

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    with tf.name_scope('maxpool5'):
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    with tf.name_scope('fc6'):
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        # tf.summary.histogram('fc6',fc6)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier6'):
        classifier6 = vernier_classifier(fc6, is_training, N_HIDDEN, name='classifier6')
        x_entropy6 = vernier_x_entropy(classifier6,y)
        correct_mean6 = vernier_correct_mean(tf.argmax(classifier6, axis=1), y)  # match correct prediction to each entry in y
        train_op6 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy6,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier6'),
                                                       name="training_op")

    # fc7
    # fc(4096, name='fc7')
    with tf.name_scope('fc7'):
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        # tf.summary.histogram('fc7', fc7)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier7'):
        classifier7 = vernier_classifier(fc7, is_training, N_HIDDEN, name='classifier7')
        x_entropy7 = vernier_x_entropy(classifier7,y)
        correct_mean7 = vernier_correct_mean(tf.argmax(classifier7, axis=1), y)  # match correct prediction to each entry in y
        train_op7 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy7,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier7'),
                                                       name="training_op")

    # fc8
    # fc(1000, relu=False, name='fc8')
    with tf.name_scope('fc8'):
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        # tf.summary.histogram('fc8', fc8)

    with tf.variable_scope('decode_vernier8'):
        classifier8 = vernier_classifier(fc8, is_training, N_HIDDEN, name='classifier8')
        x_entropy8 = vernier_x_entropy(classifier8,y)
        correct_mean8 = vernier_correct_mean(tf.argmax(classifier8, axis=1), y)  # match correct prediction to each entry in y
        train_op8 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy8,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier8'),
                                                       name="training_op")

    # prob
    # softmax(name='prob'))
    with tf.name_scope('prob'):
        prob = tf.nn.softmax(fc8)
        #tf.summary.histogram('prob',prob)

    with tf.variable_scope('decode_vernier_prob'):
        classifier_prob = vernier_classifier(prob, is_training, N_HIDDEN, name='classifier_prob')
        x_entropy_prob = vernier_x_entropy(classifier_prob,y)
        correct_mean_prob = vernier_correct_mean(tf.argmax(classifier_prob, axis=1), y)  # match correct prediction to each entry in y
        train_op_prob = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(x_entropy_prob,
                                                           var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier_prob'),
                                                           name="training_op")


    ####################################################################################################################
    # Training
    ####################################################################################################################

    if TRAINING is True:

        # training parameters
        saver = tf.train.Saver()
        # saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier'))
        init = tf.global_variables_initializer()
        summary = tf.summary.merge_all()
        update_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        master_training_op = [train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7, train_op8, train_op_prob, update_batch_norm_ops]

        with tf.Session() as sess:

            print('Training ' + STIM + '...')
            writer = tf.summary.FileWriter(LOGDIR+'/'+STIM+'_training', sess.graph)

            if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
                saver.restore(sess, checkpoint_path)
                print('Checkpoint found, will not bother training.')
            if (restore_checkpoint and tf.train.checkpoint_exists(
                    checkpoint_path) and continue_training_from_checkpoint) \
                    or not restore_checkpoint or not tf.train.checkpoint_exists(checkpoint_path):

                init.run()

                for iteration in range(n_batches):

                    # get data in the batches
                    createPatches(nSamples=batch_size//2, stimType=STIM)
                    batch_data, batch_labels = make_dataset_from_patch(patch_folder=STIM, n_repeats=1,
                                                                       resize_factor=resize_factor, print_shapes=False)

                    if iteration % 5 == 0:

                        # Run the training operation, measure the losses and write summary:
                        _, summ = sess.run(
                            [master_training_op, summary],
                            feed_dict={x: batch_data,
                                       y: batch_labels,
                                       is_training: TRAINING})
                        writer.add_summary(summ, iteration)

                    else:

                        # Run the training operation and measure the losses:
                        _ = sess.run(master_training_op,
                            feed_dict={x: batch_data,
                                       y: batch_labels,
                                       is_training: TRAINING})

                    print("\rIteration: {}/{} ({:.1f}%)".format(
                        iteration, n_batches,
                        iteration * 100 / n_batches),
                        end="")

                # save the model at the end
                save_path = saver.save(sess, checkpoint_path)


    ####################################################################################################################
    # Testing
    ####################################################################################################################


    if TRAINING is False:

        # testing parameters
        saver = tf.train.Saver()
        #saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier'))
        summary = tf.summary.merge_all()

        with tf.Session() as sess:

            print('Testing '+STIM+'...')
            writer = tf.summary.FileWriter(LOGDIR+'/'+STIM+'_testing', sess.graph)
            saver.restore(sess, checkpoint_path)

            # we will collect correct responses here: one entry per vernier decoder
            correct_responses = np.zeros(shape=(9))
            # assemble the number of correct responses for each vernier decoder
            correct_mean_all = tf.stack([correct_mean1, correct_mean2, correct_mean3, correct_mean4, correct_mean5, correct_mean6, correct_mean7,
                             correct_mean8, correct_mean_prob],axis=0, name='correct_mean_all')


            for iteration in range(n_batches):

                # get data in the batches
                createPatches(nSamples=batch_size//2, stimType=STIM)
                batch_data, batch_labels = make_dataset_from_patch(patch_folder=STIM, n_repeats=1,
                                                                   resize_factor=resize_factor, print_shapes=False)

                if iteration % 5 == 0:

                    # Run the training operation, measure the losses and write summary:
                    correct_in_this_batch_all, summ = sess.run([correct_mean_all, summary],
                                                               feed_dict={x: batch_data,
                                                                          y: batch_labels,
                                                                is_training: TRAINING})
                    writer.add_summary(summ, iteration)

                else:

                    # Run the training operation and measure the losses:
                    correct_in_this_batch_all = sess.run(correct_mean_all,
                                                         feed_dict={x: batch_data,
                                                                    y: batch_labels,
                                                                    is_training: TRAINING})

                correct_responses += np.array(correct_in_this_batch_all)

                print("\rIteration: {}/{} ({:.1f}%)".format(
                    iteration, n_batches,
                    iteration * 100 / n_batches),
                    end="")

        percent_correct = correct_responses*100/n_batches
        print('... testing done.')
        print('Percent correct for vernier decoders in ascending order: ')
        print(percent_correct)
        np.save(LOGDIR+'/'+STIM+'_percent_correct', percent_correct)

    ####################################################################################################################
    # in case we wonder what the output of alexnet itself is.
    ####################################################################################################################
    #Output:


    # for input_im_ind in range(output.shape[0]):
    #     inds = argsort(output)[input_im_ind,:]
    #     print("Image", input_im_ind)
    #     for i in range(5):
    #         print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])










def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def batch_norm_layer(x, n_out, phase, name='', activation=None):
    with tf.variable_scope('batch_norm_layer'):
        h1 = tf.layers.dense(x, n_out, activation=None, name=name)
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope=name+'bn')
    if activation is None:
        return h2
    else:
        return activation(h2)

def vernier_classifier(input, is_training, n_hidden=1024, name=''):
    with tf.name_scope(name):
        batch_size = tf.shape(input)[0]

        # find how many units are in this layer to flatten it
        items_to_multiply = len(np.shape(input))-1
        n_units = 1
        for i in range(1, items_to_multiply+1):
            n_units = n_units*int(np.shape(input)[i])

        flat_input = tf.reshape(input, [batch_size, n_units])
        tf.summary.histogram('classifier_input_no_bn', flat_input)

        flat_input = tf.contrib.layers.batch_norm(flat_input, center=True, scale=True, is_training=is_training,
                                                  scope=name + 'input_bn')
        tf.summary.histogram('classifier_input_bn', flat_input)

        if n_hidden is None:
            classifier_fc = tf.layers.dense(flat_input, 2, name='classifier_top_fc')
            # classifier_fc = batch_norm_layer(flat_input, 2, is_training, name='classifier_fc')
            tf.summary.histogram(name+'_fc', classifier_fc)
        else:
            with tf.device('/cpu:0'):
                classifier_hidden = tf.layers.dense(flat_input, n_hidden, activation=tf.nn.elu, name=name+'_hidden_fc')
                # classifier_hidden = batch_norm_layer(flat_input, n_hidden, is_training,
                # activation=tf.nn.relu, name='classifier_hidden_fc')
                tf.summary.histogram(name+'_hidden', classifier_hidden)
            classifier_fc = tf.layers.dense(classifier_hidden, 2, activation=tf.nn.elu, name=name+'_top_fc')
            # classifier_fc = batch_norm_layer(classifier_hidden, 2, is_training, name='classifier_top_fc')
            tf.summary.histogram(name+'_fc', classifier_fc)
            
        classifier_out = tf.nn.softmax(classifier_fc, name='softmax')
        
        return classifier_out


def vernier_x_entropy(prediction_vector, label):
    with tf.name_scope("x_entropy"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction_vector, labels=tf.one_hot(label,2)), name="xent")
        tf.summary.scalar("xent", xent)
        return xent


def vernier_correct_mean(prediction, label):
    with tf.name_scope('correct_mean'):
        correct = tf.equal(prediction, label, name="correct")
        correct_mean = tf.reduce_mean(tf.cast(correct, tf.float32), name="correct_mean")
        tf.summary.scalar('correct_mean', correct_mean)
        return correct_mean


if __name__=="__main__":
   main()