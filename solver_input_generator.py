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

import math

from third_party.modified.karel_env.input_ops_karel import create_test_input
from third_party.modified.vizdoom_env.input_ops_vizdoom import create_test_input_vizdoom

import numpy as np


class Solver_Input:

    def __init__(self, config, dataset, session, model, dsl, i):

        self.config = config
        self.dataset = dataset
        self.session = session
        self.model = model
        self.dsl = dsl

        print("\nProgram", i)

        if self.config.dataset_type == 'karel':

            test_input = create_test_input(self.dataset, i)

            end_token = 5

        else:

            test_input = create_test_input_vizdoom(self.dataset, i)

            end_token = 11

        [self.action_tokens, self.perception_tokens, self.action_prob, self.perception_prob] = \
            self.get_model_predictions(test_input)

        demonstrations = test_input["s_h"][0]

        program_tokens = test_input["program_tokens"][0][:int(test_input["program_len"][0])]
        program_code = self.dsl.intseq2str(program_tokens)

        action_tokens_gt = test_input["a_h_tokens"][0]
        perception_tokens_gt = np.vectorize(self.convert)(test_input["per"][0])

        test_action_tokens = test_input["test_a_h_tokens"][0]
        test_perception_tokens = np.vectorize(self.convert)(test_input["test_per"][0])

        limit = np.argwhere(self.action_tokens == end_token)[:, 1]
        test_limit = np.argwhere(test_action_tokens == end_token)[:, 1]

        print("Program", program_code)

        self.program_code = program_code
        self.action_tokens_gt = action_tokens_gt
        self.perception_tokens_gt = perception_tokens_gt
        self.test_action_tokens = test_action_tokens
        self.test_perception_tokens = test_perception_tokens
        self.limit = limit
        self.test_limit = test_limit
        self.demonstrations = demonstrations

    def convert(self,el):
        if el:
            return "#t"
        else:
            return "#f"

    def softmax(self, x, axis=4):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis),axis=axis)

    # Get prediction ah tokens with neural network
    def get_model_predictions(self, batch_chunk):

        [action_logit,perception_logit] = self.session.run([self.model.greedy_pred_action_list,
                                  self.model.greedy_pred_per_list],
                                  feed_dict=self.model.get_feed_dict(batch_chunk))



        action_tokens = np.argmax(np.array(action_logit), axis=2)[:,0]
        action_prob = self.softmax(np.array(action_logit),axis=2)[:,0]
        perception_tokens = np.vectorize(self.convert)(
            np.argmax(np.array(perception_logit),axis=2)[:,0])
        perception_prob = self.softmax(np.array(perception_logit), axis=2)[:,0]

        return [action_tokens,perception_tokens,action_prob,perception_prob]


    def sanity_check(self):

        test = self.limit.shape[0] == self.action_tokens.shape[0]

        return test

    def truncate_input(self):

        k = self.action_tokens.shape[0]
        k_test = self.test_action_tokens.shape[0]

        self.action_tokens = [self.action_tokens[j, :self.limit[j]].tolist() for j in range(k)]

        self.action_tokens_gt = [self.action_tokens_gt[j, :self.limit[j]].tolist() for j in range(k)]
        self.perception_tokens = [self.perception_tokens[j, :self.limit[j] + 1].tolist() for j in range(k)]
        self.perception_tokens_gt = [self.perception_tokens_gt[j, :self.limit[j] + 1].tolist() for j in range(k)]

        self.test_action_tokens = [self.test_action_tokens[j, :self.test_limit[j]].tolist() for j in range(k_test)]
        self.test_perception_tokens = [self.test_perception_tokens[j, :self.test_limit[j] + 1].tolist() for j in range(k_test)]

        wrong_actions = self.count_wrong_action_tokens(self.action_tokens,self.action_tokens_gt)
        wrong_perceptions = self.count_wrong_perception_tokens(self.perception_tokens,self.perception_tokens_gt)

        print("Inital wrong actions",wrong_actions,
              "Initial wrong perceptions",wrong_perceptions)

        return wrong_actions, wrong_perceptions

    def count_wrong_action_tokens(self, action_tokens, action_tokens_gt):

        sum = 0
        for i in range(len(action_tokens)):

            sum += np.sum(action_tokens[i] != action_tokens_gt[i])

        return sum

    def count_wrong_perception_tokens(self, perception_tokens, perception_tokens_gt):

        sum = 0
        for i in range(len(perception_tokens)):
            sum += np.sum(perception_tokens[i] != perception_tokens_gt[i])

        return sum

    def convert_back(self, boolean):

        if boolean == "#t":
            return 1
        return 0

    def test_perception_prediction(self, i , j, indexes, threshold):

        for l in range(self.config.per_dim):

            token = self.convert_back(self.perception_tokens[i][j][l])

            if self.perception_prob[i, token, j,l] <= threshold:

                if len(indexes) == 0 or indexes[-1] != i:
                    indexes.append(i)

    def test_action_prediction(self,i,j,indexes, threshold):

        if self.action_prob[i, self.action_tokens[i][j], j] <= threshold:

            if len(indexes) == 0 or indexes[-1] != i:
                indexes.append(i)


    def count_wrong_tokens(self, ah_tokens, ah_tokens_gt, per_tokens, per_tokens_gt):

        print("Remaining wrong actions", self.count_wrong_action_tokens(ah_tokens,
                                                              ah_tokens_gt),
              "Remaining wrong perceptions", self.count_wrong_perception_tokens(per_tokens,
                                                                per_tokens_gt),
              "Remaining demonstrations", len(ah_tokens))

    def filter_perception_threshold(self, perception_prob_threshold):

        print("Static filtering of perceptions with threshold", perception_prob_threshold)

        indexes = []

        for i in range(len(self.action_tokens)):

            action_len = len(self.action_tokens[i])

            for j in range(action_len):
                self.test_perception_prediction(i, j, indexes, perception_prob_threshold)

            self.test_perception_prediction(i, action_len, indexes, perception_prob_threshold)

        action_tokens = self.action_tokens[:]
        action_tokens_gt = self.action_tokens_gt[:]
        perception_tokens = self.perception_tokens[:]
        perception_tokens_gt = self.perception_tokens_gt[:]

        for index in sorted(indexes, reverse=True):
            del action_tokens[index]
            del action_tokens_gt[index]
            del perception_tokens[index]
            del perception_tokens_gt[index]

        self.count_wrong_tokens(self.action_tokens,
                                self.action_tokens_gt,
                                self.perception_tokens,
                                self.perception_tokens_gt)

        return action_tokens, action_tokens_gt, perception_tokens, perception_tokens_gt

    def filter_perception_proportion(self,perception_proportion):

        print("Dynamic filtering of perceptions with proportion", perception_proportion)

        n_demonstrations = len(self.action_tokens)

        number_keep = int(math.ceil(perception_proportion*n_demonstrations))
        number_remove = n_demonstrations - number_keep

        per_confidences = [1.]*n_demonstrations

        for i in range(len(self.action_tokens)):
            action_len = len(self.action_tokens[i])
            for j in range(action_len+1):
                for l in range(self.config.per_dim):
                    token = self.convert_back(self.perception_tokens[i][j][l])
                    per_confidences[i] = min(self.perception_prob[i, token, j, l],per_confidences[i])

        indexes = np.argpartition(np.array(per_confidences),number_remove)[:number_remove]

        action_tokens = self.action_tokens[:]
        action_tokens_gt = self.action_tokens_gt[:]
        perception_tokens = self.perception_tokens[:]
        perception_tokens_gt = self.perception_tokens_gt[:]

        for index in sorted(indexes, reverse=True):
            del action_tokens[index]
            del action_tokens_gt[index]
            del perception_tokens[index]
            del perception_tokens_gt[index]

        self.count_wrong_tokens(action_tokens,
                                action_tokens_gt,
                                perception_tokens,
                                perception_tokens_gt)

        return action_tokens, action_tokens_gt, perception_tokens, perception_tokens_gt


    def filter_action_threshold(self, action_prob_threshold):

        print("Static filtering of actions with threshold", action_prob_threshold)

        indexes = []

        for i in range(len(self.action_tokens)):

            action_len = len(self.action_tokens[i])

            for j in range(action_len):

                self.test_action_prediction(i, j, indexes, action_prob_threshold)

        for index in sorted(indexes, reverse=True):
            del self.action_tokens[index]
            del self.action_tokens_gt[index]
            del self.perception_tokens[index]
            del self.perception_tokens_gt[index]
            self.perception_prob = np.delete(self.perception_prob,index,axis=0)

        self.count_wrong_tokens(self.action_tokens,
                                self.action_tokens_gt,
                                self.perception_tokens,
                                self.perception_tokens_gt)

