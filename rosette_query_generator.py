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

import subprocess
import os

DEPTH = 20

class QueryGenerator():

    def __init__(self, config, exp_id):

        self.folder = os.path.join("rosette/generated_files/",exp_id)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.solve_folder = os.path.join(self.folder, "solve/")
        self.test_folder = os.path.join(self.folder ,"test/")

        if not os.path.exists(self.solve_folder):
            os.makedirs(self.solve_folder)
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)

        self.config = config

    # Recursive function to write list
    def write_list(self,f,l):
        if isinstance(l,list):
            f.write("(list ")
            for el in l:
                self.write_list(f,el)
            f.write(")\n")

        else:
            f.write(str(l)+" ")

    def generate_solver_file(self,rosette_file,program_code,ah_tokens,perceptions,synthax_file):

        with open(rosette_file, 'w') as f:
            f.write("#lang rosette/safe\n")
            f.write("(require rosette/lib/synthax)\n")
            f.write("(require racket/include)\n")


            f.write("(include  \"../../../vizdoom_synthax/constructs.rkt\")\n")
            f.write("(include \"../../../"+synthax_file+".rkt\")\n")
            f.write(";Code is " + program_code + "\n")

            if self.config.dataset_type == 'karel':
                f.write(";Perceptions are front, left, right, marker, no_marker\n")

                f.write(";Actions are move, left, right, pick, put\n")
            else:
                f.write(";Perceptions are isThere demon, isThere knight, isThere revenant, target demon, target knight, target revenant,\n")

                f.write(";Actions are  forward, backward, left, right, turn left, turn right, attack, selectw1, selectw2 , selectw3,selectw4 ,selectw5\n")

            f.write("(define actions ")
            self.write_list(f, ah_tokens)
            f.write(")\n")

            f.write("(define perceptions ")
            self.write_list(f, perceptions)
            f.write(")\n")

            f.write("(define(a_trous_program perceptions)\n")

            f.write("\t(Block (list) perceptions ))\n")

            f.write("(define binding\n")
            f.write("\t(synthesize  #:forall (list )\n")
            f.write("\t\t#:guarantee (assert\n")
            f.write("\t\t\t(and\n")
            for j in range(len(ah_tokens)):
                f.write("\t\t\t\t(equal? (a_trous_program " + \
                        "(list-ref perceptions " + str(j) + "))" + \
                        "(list-ref actions " + str(j) + "))\n")
            f.write("))))\n")
            f.write("(print-forms binding)")

    def generate_test_file(self,test_file,program_code,ah_tokens_1,test_perceptions,result,synthax_file):
        with open(test_file, 'w') as f:
            f.write("#lang rosette/safe\n")
            f.write("(require rosette/lib/synthax)\n")
            f.write("(require racket/include)\n")

            f.write("(include  \"../../../vizdoom_synthax/constructs.rkt\")\n")
            f.write("(include \"../../../" + synthax_file + ".rkt\")\n")

            f.write(";Code is " + program_code + "\n")
            f.write(";Perceptions are front, left, right, marker, no_marker\n")

            f.write(";Actions are move, left, right, pick, put\n")

            f.write("(define test_actions ")
            self.write_list(f, ah_tokens_1)
            f.write(")\n")

            f.write("(define test_perceptions ")
            self.write_list(f, test_perceptions)
            f.write(")\n")

            f.write(result)

            for j in range(len(ah_tokens_1)):
                f.write("\n(equal? (a_trous_program " + \
                        "(list-ref test_perceptions " + str(j) + "))" + \
                        "(list-ref test_actions " + str(j) + "))\n")

    def attempt(self,i,program_code,ah_tokens,perceptions,synthax_file):

        rosette_file = os.path.join(self.solve_folder,
                                    "program_" + str(i) + "_"+synthax_file.split("/")[-1] + ".rkt")
        self.generate_solver_file(rosette_file, program_code, ah_tokens, perceptions, synthax_file)
        result = "\n".join(subprocess.check_output(["racket", rosette_file]). \
                           splitlines()[1:])[1:]

        return result

    def test(self, i, program_code, ah_tokens_1, test_perceptions, synthax_file, result):

        test_file = os.path.join(self.test_folder,
                                 "program_" + str(i) + "_" + synthax_file.split("/")[-1] + ".rkt")
        self.generate_test_file(test_file, program_code, ah_tokens_1, test_perceptions, result,
                                                synthax_file)

        # Run and get solution

        second_result = subprocess.check_output(["racket", test_file]).splitlines()

        return second_result