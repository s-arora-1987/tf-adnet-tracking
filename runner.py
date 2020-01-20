from __future__ import division

import os
import random
import sys
import logging

import cv2
import fire
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.slim as slim
import commons
from boundingbox import BoundingBox, Coordinate
from configs import ADNetConf
from networks import ADNetwork
from pystopwatch import StopWatchManager

_log_level = logging.DEBUG
_logger = logging.getLogger('ADNetRunner')
_logger.setLevel(_log_level)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(_log_level)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
_logger.addHandler(ch)


class ADNetRunner:
# class for constructing ADNet using a network pre-trained in MATLAB code at https://github.com/hellbell/ADNet and running online tracking and training on it

    MAX_BATCHSIZE = 100  # 512 

    def __init__(self):
        self.tensor_input = tf.placeholder(tf.float32, shape=(None, 112, 112, 3), name='patch')
        self.tensor_action_history = tf.placeholder(tf.float32, shape=(None, 1, 1, 110), name='action_history')
        self.tensor_lb_action = tf.placeholder(tf.int32, shape=(None, ), name='lb_action')
        self.tensor_lb_class = tf.placeholder(tf.int32, shape=(None, ), name='lb_class')
        self.tensor_is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        self.roi_pool=tf.placeholder(tf.bool,name="roi_pool_switch")

        self.persistent_sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=3,
            intra_op_parallelism_threads=3
        ))
	# session for original ADNet (persistent_sess2 is similarly defined on second tf DAG graph later in this code in method roi_net_newgraph()
        self.adnet = ADNetwork(self.learning_rate_placeholder)
	# constructing ADNet using a pre-trained network 
        self.adnet.create_network(self.tensor_input, self.tensor_lb_action, self.tensor_lb_class, self.tensor_action_history, self.tensor_is_training,roi=self.roi_pool)
	# function implementing re-detection when the network fails to detect with 70% confidence 
        self.callback_redetection = self.redetection_by_sampling
        self.g_redetect = tf.Graph()

	# class instance to enable the constructing second network
        self.adnet2 = ADNetwork(self.learning_rate_placeholder)

        self.roi_net_newgraph()  # method constructing second network which implements region of interest pooling, and creating corresponding tensorflow session to run that network 
        self.use_roi = True # False for trying the original ADNet, and True for RoI-ADNet



    def by_dataset(self, vid_path='./data/freeman1/'):
        assert os.path.exists(vid_path)
	# collecting training data 
        gt_boxes = BoundingBox.read_vid_gt(vid_path)
        curr_bbox = None
	# thresholds on IoU (intersection of union) for computing success rates
        th_list = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
	self.stopwatch = StopWatchManager()
	# file where accuracy and time of multiple runs were recorded.
	# if you wish to test results of multiple runs, please change path of this file according to your system directory
        #f = open("/home/saurabharora/output.txt", "a") 
        import time

	if 'ADNET_MODEL_PATH' in os.environ.keys():
		self.adnet.read_original_weights(self.persistent_sess, os.environ['ADNET_MODEL_PATH'])
	else:
		self.adnet.read_original_weights(self.persistent_sess)

	self.action_histories = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
	self.action_histories_old = np.array([0] * ADNetConf.get()['action_history'], dtype=np.int8)
	self.histories = []
	self.iteration = 0
	self.imgwh = None
	self.failed_cnt = 0
	self.latest_score = 0
	# list storing values of IoU between ground truth and predicted bounding boxes
	IoU_list = []
	success_rates = np.zeros(len(th_list))
	self.stopwatch.start('total')
	tm = time.time()
	_logger.info('---- start dataset l=%d' % (len(gt_boxes)))

	for idx, gt_box in enumerate(gt_boxes):
		img = commons.imread(os.path.join(vid_path, 'img', '%04d.jpg' % (idx + 1)))
		self.imgwh = Coordinate.get_imgwh(img)
		if idx == 0:
		    # initialization : initial fine-tuning
		    self.initial_finetune(img, gt_box)
		    curr_bbox = gt_box
		# tracking
		predicted_box = self.tracking(img, curr_bbox)
		self.show(img, gt_box=gt_box, predicted_box=predicted_box)
		# compute and add IoU value to list
		IoU_list.append(gt_box.iou(predicted_box))
		# cv2.imwrite('/Users/ildoonet/Downloads/aaa/%d.jpg' % self.iteration, img)
		curr_bbox = predicted_box

	self.stopwatch.stop('total')
	elapsed = time.time() - tm
	_logger.info('----')
	_logger.info(self.stopwatch)
	_logger.info('%.3f FPS' % (len(gt_boxes) / self.stopwatch.get_elapsed('total')))
	# write time data to file
	#f.write(str(self.stopwatch))
	#f.write("\n")

	for i in range(len(th_list)):
	# if the 
		success_rates[i] = 0
		for iou in IoU_list:
			if iou > th_list[i]:
				success_rates[i] += 1
		i += 1

	#print success_rates
	#f.write(str(success_rates))
	#f.write("\n")

	# computing success rate as a percentage of the successes for 0 threshold on IoU 
	maxframes = success_rates[0]
	for i in range(len(success_rates)):
		success_rates[i] = success_rates[i]*100/maxframes
		#f.write("\nSR"+str(i+1)+":\n"+str(success_rates[i]))
	
	print "Success rates for this run:"
	print success_rates
	#f.write("\nLD:\n"+str(elapsed)+"\n")

        #f.close()


    def show(self, img, delay=1, predicted_box=None, gt_box=None):
	# show ground truth and prediction
        if isinstance(img, str):
            img = commons.imread(img)

        if gt_box is not None:
            gt_box.draw(img, BoundingBox.COLOR_GT)
        if predicted_box is not None:
            predicted_box.draw(img, BoundingBox.COLOR_PREDICT)

        cv2.imshow('result', img)
        cv2.waitKey(delay)

    def _get_features(self, samples):
	# compute feature map for input samples 
        feats = []
        for batch in commons.chunker(samples, ADNetRunner.MAX_BATCHSIZE):
            feats_batch = self.persistent_sess.run(self.adnet.layer_feat, feed_dict={
                self.adnet.input_tensor: batch,
                self.roi_pool: False
            })
            feats.extend(feats_batch)
        return feats

    def initial_finetune(self, img, detection_box):
        self.stopwatch.start('initial_finetune')
        t = time.time()

        # generate samples
        pos_num, neg_num = ADNetConf.g()['initial_finetune']['pos_num'], ADNetConf.g()['initial_finetune']['neg_num']
        pos_boxes, neg_boxes = detection_box.get_posneg_samples(self.imgwh, pos_num, neg_num, use_whole=True)
        pos_lb_action = BoundingBox.get_action_labels(pos_boxes, detection_box)

        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
        for box, feat in zip(pos_boxes, feats):
            box.feat = feat
        feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
        for box, feat in zip(neg_boxes, feats):
            box.feat = feat

        # train_fc_finetune_hem
        self._finetune_fc(
            img, pos_boxes, neg_boxes, pos_lb_action,
            ADNetConf.get()['initial_finetune']['learning_rate'],
            ADNetConf.get()['initial_finetune']['iter']
        )

        self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))
        _logger.info('ADNetRunner.initial_finetune t=%.3f' % t)
        self.stopwatch.stop('initial_finetune')

    def _finetune_fc(self, img, pos_boxes, neg_boxes, pos_lb_action, learning_rate, iter, iter_score=1):
        BATCHSIZE = ADNetConf.g()['minibatch_size']

        def get_img(idx, posneg):
            if isinstance(img, tuple):
                return img[posneg][idx]
            return img

        pos_samples = [commons.extract_region(get_img(i, 0), box) for i, box in enumerate(pos_boxes)]
        neg_samples = [commons.extract_region(get_img(i, 1), box) for i, box in enumerate(neg_boxes)]
        # pos_feats, neg_feats = self._get_features(pos_samples), self._get_features(neg_samples)

        # commons.imshow_grid('pos', pos_samples[-50:], 10, 5)
        # commons.imshow_grid('neg', neg_samples[-50:], 10, 5)
        # cv2.waitKey(1)

        for i in range(iter):
            batch_idxs = commons.random_idxs(len(pos_boxes), BATCHSIZE)
            batch_feats = [x.feat for x in commons.choices_by_idx(pos_boxes, batch_idxs)]
            batch_lb_action = commons.choices_by_idx(pos_lb_action, batch_idxs)
            self.persistent_sess.run(
                self.adnet.weighted_grads_op1,
                feed_dict={
                    self.adnet.layer_feat: batch_feats,
                    self.adnet.label_tensor: batch_lb_action,
                    self.adnet.action_history_tensor: np.zeros(shape=(BATCHSIZE, 1, 1, 110)),
                    self.learning_rate_placeholder: learning_rate,
                    self.tensor_is_training: True,
                    self.roi_pool: False
                }
            )

            if i % iter_score == 0:
                # training score auxiliary(fc2)
                # -- hard score example mining
                scores = []
                for batch_neg in commons.chunker([x.feat for x in neg_boxes], ADNetRunner.MAX_BATCHSIZE):
                    scores_batch = self.persistent_sess.run(
                        self.adnet.layer_scores,
                        feed_dict={
                            self.adnet.layer_feat: batch_neg,
                            self.adnet.action_history_tensor: np.zeros(shape=(len(batch_neg), 1, 1, 110)),
                            self.learning_rate_placeholder: learning_rate,
                            self.tensor_is_training: False,
                            self.roi_pool: False
                        }
                    )
                    scores.extend(scores_batch)
                desc_order_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x:x[1][1])]

                # -- train
                batch_feats_neg = [x.feat for x in commons.choices_by_idx(neg_boxes, desc_order_idx[:BATCHSIZE])]
                self.persistent_sess.run(
                    self.adnet.weighted_grads_op2,
                    feed_dict={
                        self.adnet.layer_feat: batch_feats + batch_feats_neg,
                        self.adnet.class_tensor: [1]*len(batch_feats) + [0]*len(batch_feats_neg),
                        self.adnet.action_history_tensor: np.zeros(shape=(len(batch_feats)+len(batch_feats_neg), 1, 1, 110)),
                        self.learning_rate_placeholder: learning_rate,
                        self.tensor_is_training: True,
                        self.roi_pool: False

                    }
                )

    def tracking(self, img, curr_bbox):
        self.iteration += 1
        is_tracked = True
        boxes = []
        self.latest_score = -1
        self.stopwatch.start('tracking.do_action')
        for track_i in range(ADNetConf.get()['predict']['num_action']):
            patch = commons.extract_region(img, curr_bbox)

            # forward with image & action history
            actions, classes = self.persistent_sess.run(
                [self.adnet.layer_actions, self.adnet.layer_scores],
                feed_dict={
                    self.adnet.input_tensor: [patch],
                    self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories)],
                    self.tensor_is_training: False,
                    self.roi_pool: False
                }
            )

            latest_score = classes[0][1]
            if latest_score < ADNetConf.g()['predict']['thresh_fail']:
                is_tracked = False
                self.action_histories_old = np.copy(self.action_histories)
                self.action_histories = np.insert(self.action_histories, 0, 12)[:-1]
                break
            else:
                self.failed_cnt = 0
            self.latest_score = latest_score

            # move box
            action_idx = np.argmax(actions[0])
            self.action_histories = np.insert(self.action_histories, 0, action_idx)[:-1]
            prev_bbox = curr_bbox
            curr_bbox = curr_bbox.do_action(self.imgwh, action_idx)
            if action_idx != ADNetwork.ACTION_IDX_STOP:
                if prev_bbox == curr_bbox:
                    print('action idx', action_idx)
                    print(prev_bbox)
                    print(curr_bbox)
                    raise Exception('box not moved.')

            # check if consecutive actions keep predicted box oscillating between states
            if action_idx != ADNetwork.ACTION_IDX_STOP and curr_bbox in boxes:
                action_idx = ADNetwork.ACTION_IDX_STOP

            if action_idx == ADNetwork.ACTION_IDX_STOP:
                break

            boxes.append(curr_bbox)
        self.stopwatch.stop('tracking.do_action')

        # redetection when tracking failed
        new_score = 0.0
        if not is_tracked:
            self.failed_cnt += 1
            # run redetection callback function
            new_box, new_score = self.callback_redetection(curr_bbox, img)
            if new_box is not None:
                curr_bbox = new_box
                patch = commons.extract_region(img, curr_bbox)
            _logger.debug('redetection success=%s' % (str(new_box is not None)))

        # save samples
        if is_tracked or new_score > ADNetConf.g()['predict']['thresh_success']:
            self.stopwatch.start('tracking.save_samples.roi')
            imgwh = Coordinate.get_imgwh(img)
            pos_num, neg_num = ADNetConf.g()['finetune']['pos_num'], ADNetConf.g()['finetune']['neg_num']
            pos_boxes, neg_boxes = curr_bbox.get_posneg_samples(
                imgwh, pos_num, neg_num, use_whole=False,
                pos_thresh=ADNetConf.g()['finetune']['pos_thresh'],
                neg_thresh=ADNetConf.g()['finetune']['neg_thresh'],
                uniform_translation_f=2,
                uniform_scale_f=5
            )
            self.stopwatch.stop('tracking.save_samples.roi')
            self.stopwatch.start('tracking.save_samples.feat')
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(pos_boxes)])
            for box, feat in zip(pos_boxes, feats):
                box.feat = feat
            feats = self._get_features([commons.extract_region(img, box) for i, box in enumerate(neg_boxes)])
            for box, feat in zip(neg_boxes, feats):
                box.feat = feat
            pos_lb_action = BoundingBox.get_action_labels(pos_boxes, curr_bbox)
            self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))

            # clear old ones
            self.histories = self.histories[-ADNetConf.g()['finetune']['long_term']:]
            self.stopwatch.stop('tracking.save_samples.feat')

        # online finetune
        if self.iteration % ADNetConf.g()['finetune']['interval'] == 0 or is_tracked is False:
            img_pos, img_neg = [], []
            pos_boxes, neg_boxes, pos_lb_action = [], [], []
            pos_term = 'long_term' if is_tracked else 'short_term'
            for i in range(ADNetConf.g()['finetune'][pos_term]):
                if i >= len(self.histories):
                    break
                pos_boxes.extend(self.histories[-(i+1)][0])
                pos_lb_action.extend(self.histories[-(i+1)][2])
                img_pos.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][0]))
            for i in range(ADNetConf.g()['finetune']['short_term']):
                if i >= len(self.histories):
                    break
                neg_boxes.extend(self.histories[-(i+1)][1])
                img_neg.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][1]))
            self.stopwatch.start('tracking.online_finetune')
            self._finetune_fc(
                (img_pos, img_neg), pos_boxes, neg_boxes, pos_lb_action,
                ADNetConf.get()['finetune']['learning_rate'],
                ADNetConf.get()['finetune']['iter']
            )
            _logger.debug('finetuned')
            self.stopwatch.stop('tracking.online_finetune')

        cv2.imshow('patch', patch)
        return curr_bbox

    def redetection_by_sampling(self, prev_box, img):
        """
        default redetection method
        """
        imgwh = Coordinate.get_imgwh(img)
        translation_f = min(1.5, 0.6 * 1.15**self.failed_cnt)
	# creating candidate boxes by adding gaussian noise to previous target bounding box stored in variable prev_box 
        candidates = prev_box.gen_noise_samples(imgwh, 'gaussian', ADNetConf.g()['redetection']['samples'],
                                                gaussian_translation_f=translation_f)

	# list for saving values of tracking confidence P(target|state) given by fc6_2 layer, for all candidates
        scores = []
        for c_batch in commons.chunker(candidates, ADNetRunner.MAX_BATCHSIZE):
            samples =[]
            for box in c_batch:
                if not (img==[]): 
		# creating patches if image is not empty
                    samples.append(commons.extract_region(img, box))

            if self.use_roi == True:
		# Using candidate patches as regions of interest and running RoI network in function roi_net_newgraph() 
                with self.g_redetect.as_default():
                    classes = self.persistent_sess2.run(
                        self.layer_scores2,
                        feed_dict={
                            self.input_tensor2: samples
                        }
                    )
            else:
		# Using candidates for running re-detection for original ADNet
                classes = self.persistent_sess.run(
                    self.adnet.layer_scores,
                    feed_dict={
                        self.adnet.input_tensor: samples,
                        self.adnet.action_history_tensor: [commons.onehot_flatten(self.action_histories_old)]*len(c_batch),
                        self.tensor_is_training: False,
                        self.roi_pool:True
                    }
                )
            scores.extend([x[1] for x in classes])

        top5_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])][:5]
        mean_score = sum([scores[x] for x in top5_idx]) / 5.0 

	# mean score is computed by top five boxes in sorted list of confidence values,
	# and if the score is greater than desired threshold, a mean value of 
	# top five bounding boxes (top-left coordinates, width and height) is returned
        if mean_score >= self.latest_score:
            mean_box = candidates[0]
            for i in range(1, 5):
                mean_box += candidates[i]
            return mean_box / 5.0, mean_score
        return None, 0.0

    def roi_net_newgraph(self):

        with self.g_redetect.as_default():
            # feature extractor - convolutions
            self.input_tensor2 = tf.placeholder(tf.float32, shape=(None, 112, 112, 3), name='patch')
            self.label_tensor2 = tf.placeholder(tf.int32, shape=(None,), name='lb_action')
            self.class_tensor2 = tf.placeholder(tf.int32, shape=(None,), name='lb_class')

            net = slim.convolution(self.input_tensor2, 96, [7, 7], 2, padding='VALID', scope='convr',
                                   activation_fn=tf.nn.relu)

            net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*5, beta=0.75)
            net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='poolr')
	    # region of interest pooling is implemented twice by using size of stride same as the size of
	    # side of pooling window: first using averaging and then maximum 
            net = slim.pool(net, [3, 3], 'AVG', stride=3, padding='VALID', scope='pool_roi_1')
            net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool_roi_2')

            # auxilaries for flattening the outputs
            out_actions = flatten_convolution(net)
            out_scores = flatten_convolution(net)
            self.layer_actions2 = tf.nn.softmax(out_actions)
            self.layer_scores2 = tf.nn.softmax(out_scores)

            # losses
            self.loss_actions2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_tensor2, logits=out_actions)
            self.loss_cls2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.class_tensor2, logits=out_scores)

            init = tf.global_variables_initializer()

        self.persistent_sess2 = tf.Session(graph=self.g_redetect)
        self.persistent_sess2.run(init)

        return

    def __del__(self):
        self.persistent_sess.close()

def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat

if __name__ == '__main__':
    ADNetConf.get('./conf/repo.yaml')

    random.seed(1258)
    np.random.seed(1258)
    tf.set_random_seed(1258)

    fire.Fire(ADNetRunner)
