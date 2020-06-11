# coding=utf-8
"""
Copyright 2020 Raphaël Dang-Nhu
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from racket_compiler.racket_parser import Parser
from solver_input_generator import Solver_Input

from third_party.demo2program.karel_env.dsl import get_KarelDSL

from third_party.demo2program.karel_env.dsl import dsl_enum_program as karel_enum
from third_party.demo2program.vizdoom_env.dsl import dsl_enum_program as vizdoom_enum

import numpy as np
import tensorflow as tf
import h5py
import pandas as pd

from rosette_query_generator import *
from model.model_ours import Model

import time

WARMUP=10

class Solver():

    def __init__(self, config, dataset, dsl=None):

        self.config = config
        self.dataset = dataset
        self.dsl = dsl
        self.batch_size = config.batch_size

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )

        self.session = tf.Session(config=session_config)

        self.parsing_class = Parser(config.dataset_type)
        self.lexer = self.parsing_class.racket_lexer
        self.parser = self.parsing_class.racket_parser

        if config.dataset_type == 'karel':

            if config.ablation_loop_heuristic:
                self.synthax_files = ["karel_synthax/0_if_0_while",
                                      "karel_synthax/0_if_1_while_length_20",
                                      "karel_synthax/1_if_0_while",
                                      "karel_synthax/0_if_2_while"]

                if config.ablation_while_heuristic:
                    raise Exception("Both ablations are uncompatible")

            elif config.ablation_while_heuristic:
                self.synthax_files = ["karel_synthax/0_if_0_while",
                                      "karel_synthax/1_if_0_while",
                                      "karel_synthax/0_if_1_while_length_0",
                                      "karel_synthax/0_if_1_while_length_1",
                                      "karel_synthax/0_if_1_while_length_2",
                                      "karel_synthax/0_if_1_while_length_3",
                                      "karel_synthax/0_if_1_while_length_4",
                                      "karel_synthax/0_if_1_while_length_5",
                                      "karel_synthax/0_if_1_while_length_6",
                                      "karel_synthax/0_if_1_while_length_7",
                                      "karel_synthax/0_if_1_while_length_20",
                                      "karel_synthax/2_if_0_while"]

            else:
                self.synthax_files = ["karel_synthax/0_if_0_while",
                                      "karel_synthax/0_if_1_while_length_0",
                                      "karel_synthax/0_if_1_while_length_1",
                                      "karel_synthax/0_if_1_while_length_2",
                                      "karel_synthax/0_if_1_while_length_3",
                                      "karel_synthax/0_if_1_while_length_4",
                                      "karel_synthax/0_if_1_while_length_5",
                                      "karel_synthax/0_if_1_while_length_6",
                                      "karel_synthax/0_if_1_while_length_7",
                                      "karel_synthax/0_if_1_while_length_20",
                                      "karel_synthax/1_if_0_while",
                                      "karel_synthax/0_if_2_while"]

            self.enum = karel_enum

            self.action_threshold = 0
            self.perception_thresholds = [0]

        else:

            self.synthax_files = ["vizdoom_synthax/0_if_0_while",
                                  "vizdoom_synthax/1_if_0_while",
                                  "vizdoom_synthax/0_if_1_while",
                                  "vizdoom_synthax/2_if_0_while"]

            if self.config.filtering == "none":
                print("No filtering")
                self.action_threshold = 0.
                self.perception_thresholds = [0.]

            elif self.config.filtering == "static":
                print("Static filtering")
                self.action_threshold = 0.98
                self.perception_thresholds = [0.9]

            else:
                print("Dynamic filtering")
                self.action_threshold = 0.98
                self.perception_thresholds = [1.,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

            self.enum = vizdoom_enum

        self.model = Model(config, is_train=False)

        self.checkpoint = config.checkpoint
        self.checkpoint_name = os.path.basename(self.checkpoint)

        self.saver = tf.train.Saver(max_to_keep=100)

        self.saver.restore(self.session, self.checkpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        exp_name = self.config.checkpoint.split("/")[-2]
        k_str = str(self.config.num_k)
        filter = self.config.filtering
        self.exp_id = exp_name + "_k_"+ k_str + "_filter_"+ filter

        self.query_generator = QueryGenerator(config,self.exp_id)


    def check_correct_program(self, result, gt_program):

        output_program = self.parser.parse(result,lexer=self.lexer)

        if output_program is None:
            return False

        print("Program")
        print(gt_program)
        print(output_program)

        p_prog, _ = self.enum.parse(output_program)
        gt_prog, _ = self.enum.parse(gt_program)

        if p_prog is None or gt_prog is None:
            print("Parsing error")
            return False,False

        return gt_program == output_program, p_prog == gt_prog

    def check_correct_execution(self, second_result):

        n_false = 0
        for l in second_result:
            if l == "#f":
                n_false += 1

        if n_false == 0:
            correct_execution = True

        else:
            correct_execution = False
            print("FAILURE - Test Set", n_false)

        return correct_execution

    def log_results(self,count,count_correct_programs,count_correct_sequence,
                    count_incorrect_specification,mean_time_inference,max_time_inference,
                    mean_time_solver,max_time_solver):

        denominator = float(len(self.dataset.ids))
        count /= denominator
        count_correct_programs /= denominator
        count_correct_sequence /= denominator
        count_incorrect_specification /= denominator
        mean_time_inference /= denominator-WARMUP
        mean_time_solver /= denominator

        data = { "Execution accuracy": count,
                 "Program accuracy": count_correct_programs,
                 "Sequence accuracy": count_correct_sequence,
                 "Mean time inference": mean_time_inference,
                 "Max time inference": max_time_inference,
                 "Mean time solver": mean_time_solver,
                 "Max time solver": max_time_solver }

        df = pd.DataFrame.from_dict(data,columns=["Value"],orient="index")

        # Save results
        log_file = "solver_logs/"+ self.exp_id+".txt"
        with open(log_file,"w") as f:
            df.to_csv(log_file,float_format="%.5f")


    def solve(self):

        count = 0
        count_correct_programs = 0
        count_correct_sequence = 0
        count_test_set_failure = 0
        count_unsat_failure = 0
        count_incorrect_specification = 0
        mean_time_inference = 0.
        max_time_inference = 0.
        mean_time_solver = 0.
        max_time_solver = 0.


        for i in range(len(self.dataset.ids)):

            time_1 = time.time()

            solver_input = Solver_Input(self.config, self.dataset, self.session, self.model, self.dsl, i)

            time_2 = time.time()
            time_inference = time_2 - time_1

            if i > WARMUP:
                mean_time_inference += time_inference
                if  time_inference > max_time_inference:
                    max_time_inference = time_inference

            max_time = 0

            if solver_input.sanity_check():

                wrong_actions, wrong_perceptions = solver_input.truncate_input()
                if wrong_actions > 0 or wrong_perceptions > 0:
                    count_incorrect_specification += 1

                solver_input.filter_action_threshold(self.action_threshold)

                for perception_threshold in self.perception_thresholds:

                    if self.config.filtering == "dynamic":
                        action_tokens, action_tokens_gt, perception_tokens, perception_tokens_gt = \
                            solver_input.filter_perception_proportion(perception_threshold)
                    else:
                        action_tokens, action_tokens_gt, perception_tokens, perception_tokens_gt = \
                            solver_input.filter_perception_threshold(perception_threshold)

                    is_break = False

                    for synthax_file in self.synthax_files:

                        print(synthax_file)

                        try:

                            time_3 = time.time()
                            result = self.query_generator.attempt(i,
                                                                  solver_input.program_code,
                                                                  action_tokens,
                                                                  perception_tokens,
                                                                  synthax_file)

                            time_4 = time.time()
                            solver_time = time_4 - time_3
                            if solver_time > max_time:
                                max_time = solver_time

                            is_break = True
                            break
                        except:
                            result = None

                            time_4 = time.time()
                            solver_time = time_4 - time_3
                            if solver_time > max_time:
                                max_time = solver_time

                    if is_break:
                        break

                if result is not None:
                    try:

                        second_result = self.query_generator.test(i,
                                                              solver_input.program_code,
                                                              solver_input.test_action_tokens,
                                                              solver_input.test_perception_tokens,
                                                              synthax_file,
                                                              result)

                        correct_execution = self.check_correct_execution(second_result)

                        correct_sequence, correct_program = self.check_correct_program(result,solver_input.program_code)

                    except:
                        print("FAILURE - Test Set Exception")
                        correct_execution = False
                        correct_sequence = False
                        correct_program = False

                    if correct_execution:
                        count += 1
                        if correct_program:
                            count_correct_programs += 1
                            if correct_sequence:
                                count_correct_sequence += 1
                                print("SUCCESS - ALL")
                            else:
                                print("SUCCESS - PROGRAM")
                        else:
                            print("SUCCESS - EXECUTION")
                    else:
                        count_test_set_failure += 1

                else:
                    print("FAILURE - Unsat")
                    count_unsat_failure += 1

            else:

                print("Failure - Input Problem")

            mean_time_solver += max_time
            if max_time_solver < max_time:
                max_time_solver = max_time

            denominator = float(i + 1)
            print("EA", count / denominator,
                  "PA", count_correct_programs / denominator,
                  "SA", count_correct_sequence / denominator)
            print("Unsat", count_unsat_failure / denominator,
                  "Test set", count_test_set_failure / denominator)
            print("Time inference", time_inference,
                  "Time solver",max_time)


        self.log_results(count,count_correct_programs,count_correct_sequence,
                         count_incorrect_specification,mean_time_inference,max_time_inference,
                         mean_time_solver,max_time_solver)

def generate_config(parser):

    config = parser.parse_args()

    if config.dataset_type == 'karel':

        # Get dsl
        f = h5py.File(os.path.join(config.dataset_path, 'data.hdf5'), 'r')
        dsl_type = f['data_info']['dsl_type'].value
        dsl = get_KarelDSL(dsl_type=dsl_type)
        f.close()

        import third_party.modified.karel_env.dataset_karel as dataset

        dataset_train, dataset_test, dataset_val \
            = dataset.create_default_splits(config.dataset_path, num_k=config.num_k)

    elif config.dataset_type == 'vizdoom':

        import third_party.modified.vizdoom_env.dataset_vizdoom as dataset

        dataset_train, dataset_test, dataset_val \
            = dataset.create_default_splits(config.dataset_path, num_k=config.num_k)

        from third_party.demo2program.vizdoom_env.dsl.vocab import VizDoomDSLVocab

        dsl = VizDoomDSLVocab(
            perception_type=dataset_test.perception_type,
            level=dataset_test.level)

    else:
        raise ValueError(config.dataset)

    config.batch_size = 1

    # Set data dimension in configuration
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

    return config, dataset_test, dsl

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_type', type=str, default='karel',
                        choices=['karel', 'vizdoom'])
    parser.add_argument('--filtering', type=str, default='none',
                        choices=['none', 'static','dynamic'])
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/karel_dataset',
                        help='the path to your dataset')
    parser.add_argument('--num_k', type=int, default=10,
                        help='the number of seen demonstrations')
    parser.add_argument('--num_lstm_cell_units', type=int, default=512)
    parser.add_argument('--checkpoint', type=str, default='',help='the path to a trained checkpoint')

    # Ablation of solver
    parser.add_argument('--ablation_loop_heuristic', action='store_true', default=False,
                        help='set to True to ablate loop heuristic on Karel')
    parser.add_argument('--ablation_while_heuristic', action='store_true', default=False,
                        help='set to True to ablate while heuristic on Karel')

    config, dataset_test, dsl = generate_config(parser)

    solver = Solver(config, dataset_test, dsl=dsl)
    solver.solve()

if __name__ == '__main__':
    main()
