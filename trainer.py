# See License in third_party/demo2program

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import os
import sys

from third_party.demo2program.models.util import log
from model.model_ours import Model


class Trainer(object):

    def __init__(self,
                 config,
                 dataset,
                 dataset_test):
        self.config = config
        hyper_parameter_str = 'bs_{}_lr_{}_{}_cell_{}'.format(
            config.batch_size, config.learning_rate,
            'lstm',
            config.num_lstm_cell_units)

        hyper_parameter_str += '_k_{}'.format(self.config.num_k)

        self.train_dir = './train_dir/'+config.output

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        if config.dataset_type == 'karel':

            from third_party.demo2program.karel_env.dsl import get_KarelDSL

            self.dsl = get_KarelDSL(dsl_type='prob')

            from third_party.demo2program.karel_env.input_ops_karel import create_input_ops

        elif config.dataset_type == 'vizdoom':

            from third_party.demo2program.vizdoom_env.dsl.vocab import VizDoomDSLVocab

            self.dsl = VizDoomDSLVocab(
                perception_type=dataset_test.perception_type,
                level=dataset_test.level)

            from third_party.demo2program.vizdoom_env.input_ops_vizdoom import create_input_ops
        else:
            raise ValueError(config.dataset)

        _, self.batch_train = create_input_ops(dataset, self.batch_size,
                                               is_training=True)
        _, self.batch_test = create_input_ops(dataset_test, self.batch_size,
                                              is_training=False)
        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(
            graph=None)

        # --- create model ---
        self.model = Model(config, debug_information=config.debug,
                           global_step=self.global_step)

        self.learning_rate = config.learning_rate

        self.check_op = tf.no_op()

        # --- checkpoint and monitoring ---
        all_vars = tf.trainable_variables()
        log.warn("********* var ********** ")
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='optimizer_pixel_loss'
        )

        self.train_summary_op = tf.summary.merge_all(key='train')
        self.test_summary_op = tf.summary.merge_all(key='test')

        self.saver = tf.train.Saver(max_to_keep=100)
        self.pretrain_saver = tf.train.Saver(var_list=all_vars, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.log_step = self.config.log_step
        self.test_sample_step = self.config.test_sample_step
        self.write_summary_step = self.config.write_summary_step

        self.checkpoint_secs = 600  # 10 min

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(
            config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.pretrain_saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided" +
                     "checkpoint path")

    def train(self):

        log.infov("Training Starts!")

        max_steps = self.config.max_steps

        ckpt_save_step = 200
        debug_step = 100
        log_step = self.log_step
        test_sample_step = self.test_sample_step
        write_summary_step = self.write_summary_step

        for s in xrange(max_steps):

            step, train_summary, loss, output, step_time = \
                self.run_single_step(
                    self.batch_train, step=s, is_train=True,debug_step=debug_step)
            if s % log_step == 0:
                self.log_step_message(step, loss, step_time)

            if s % test_sample_step == 0:
                test_step, test_summary, test_loss, output, test_step_time = \
                    self.run_test(self.batch_test)
                self.summary_writer.add_summary(test_summary,
                                                global_step=test_step)
                self.log_step_message(step, test_loss, test_step_time, is_train=False)


            if s % write_summary_step == 0:
                self.summary_writer.add_summary(train_summary,
                                                global_step=step)

            if s % ckpt_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                self.saver.save(
                    self.session, os.path.join(self.train_dir, 'model'),
                    global_step=step)

    def run_single_step(self, batch, step=None, is_train=True, debug_step=None):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.train_summary_op, self.model.output,
                 self.model.loss, self.check_op, self.optimizer,
                 self.model.greedy_pred_action_list,
                 self.model.greedy_pred_per_list,
                 self.model.action_len]

        feed_dict = self.model.get_feed_dict(
            batch_chunk, step=step,
            is_training=is_train,
        )

        fetch_values = self.session.run(fetch, feed_dict=feed_dict)
        [step, summary, output, loss] = fetch_values[:4]

        _end_time = time.time()

        return step, summary, loss, output, (_end_time - _start_time)

    def run_test(self, batch):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        feed_dict = self.model.get_feed_dict(
            batch_chunk,
            is_training=False,
        )

        step, summary, loss, output = self.session.run(
            [self.global_step, self.test_summary_op, self.model.loss,
             self.model.output],
            feed_dict=feed_dict
        )

        _end_time = time.time()

        return step, summary, loss, output, (_end_time - _start_time)

    def log_step_message(self, step, loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} " +
                "instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='set to True to see debugging visualization')
    parser.add_argument('--output', type=str, default='default',
                        help='log folder for training')

    parser.add_argument('--dataset_type', type=str, default='karel',
                        choices=['karel', 'vizdoom'])
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/karel_dataset',
                        help='the path to your dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='specify the path to a pre-trained checkpoint')
    # log
    parser.add_argument('--log_step', type=int, default=100,
                        help='the frequency of outputing log info')
    parser.add_argument('--write_summary_step', type=int, default=100,
                        help=' the frequency of writing TensorBoard sumamries')
    parser.add_argument('--test_sample_step', type=int, default=100,
                        help='the frequency of performing '
                             'testing inference during training')
    # hyperparameters
    parser.add_argument('--num_k', type=int, default=10,
                        help='the number of seen demonstrations')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_steps', type=int, default=10000)

    # model hyperparameters
    parser.add_argument('--num_lstm_cell_units', type=int, default=512)

    config = parser.parse_args()

    if config.dataset_type == 'karel':
        import third_party.modified.karel_env.dataset_karel as dataset
        dataset_train, dataset_test, dataset_val \
            = dataset.create_default_splits(config.dataset_path, num_k=config.num_k)
    elif config.dataset_type == 'vizdoom':
        import third_party.modified.vizdoom_env.dataset_vizdoom as dataset
        dataset_train, dataset_test, dataset_val \
            = dataset.create_default_splits(config.dataset_path, num_k=config.num_k)
    else:
        raise ValueError(config.dataset)

    data_tuple = dataset_train.get_data(dataset_train.ids[0])
    program, _, s_h, test_s_h, a_h, _, _, _, program_len, demo_len, test_demo_len, \
        per, test_per = data_tuple[:13]


    config.dim_program_token = np.asarray(program.shape)[0]
    config.max_program_len = np.asarray(program.shape)[1]
    config.k = np.asarray(s_h.shape)[0]
    config.test_k = np.asarray(test_s_h.shape)[0]
    config.max_demo_len = np.asarray(s_h.shape)[1]
    config.h = np.asarray(s_h.shape)[2]
    config.w = np.asarray(s_h.shape)[3]
    config.depth = np.asarray(s_h.shape)[4]
    config.action_space = np.asarray(a_h.shape)[2]
    config.per_dim = np.asarray(per.shape)[2]
    if config.dataset_type == 'karel':
        config.dsl_type = dataset_train.dsl_type
        config.env_type = dataset_train.env_type
        config.vizdoom_pos_keys = []
        config.vizdoom_max_init_pos_len = -1
        config.perception_type = ''
        config.level = None
    elif config.dataset_type == 'vizdoom':
        config.dsl_type = 'vizdoom_default'  # vizdoom has 1 dsl type for now
        config.env_type = 'vizdoom_default'  # vizdoom has 1 env type
        config.vizdoom_pos_keys = dataset_train.vizdoom_pos_keys
        config.vizdoom_max_init_pos_len = dataset_train.vizdoom_max_init_pos_len
        config.perception_type = dataset_train.perception_type
        config.level = dataset_train.level

    trainer = Trainer(config, dataset_train, dataset_test)


    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()

if __name__ == '__main__':
    main()
