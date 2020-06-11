import numpy as np
import tensorflow as tf

from third_party.demo2program.vizdoom_env.util import log

def create_test_input_vizdoom(dataset,id):

    id_string = dataset.ids[id]
    data = dataset.get_data(id_string)

    program, program_tokens, s_h, test_s_h, a_h, a_h_tokens, \
    test_a_h, test_a_h_tokens, program_len, demo_len, test_demo_len, \
    per, test_per, ip, ipl, tip, tipl = data

    input_ops = {}

    input_ops['id'], input_ops['program'], input_ops['program_tokens'], \
    input_ops['s_h'], input_ops['test_s_h'], \
    input_ops['a_h'], input_ops['a_h_tokens'], \
    input_ops['test_a_h'], input_ops['test_a_h_tokens'], \
    input_ops['program_len'], input_ops['demo_len'], \
    input_ops['test_demo_len'], input_ops['per'], input_ops['test_per'], \
    input_ops['init_pos'], input_ops['init_pos_len'], \
            input_ops['test_init_pos'], input_ops['test_init_pos_len'] =\
    (id_string, program.astype(np.float32), program_tokens.astype(np.int32),\
        s_h.astype(np.float32), test_s_h.astype(np.float32),\
        a_h.astype(np.float32), a_h_tokens.astype(np.int32),\
        test_a_h.astype(np.float32), test_a_h_tokens.astype(np.int32),\
        program_len.astype(np.float32), demo_len.astype(np.float32),\
        test_demo_len.astype(np.float32),\
        per.astype(np.float32), test_per.astype(np.float32), ip.astype(np.int32), \
        ipl.astype(np.int32), tip.astype(np.int32), \
        tipl.astype(np.int32))

    for key,item in input_ops.items():
        input_ops[key] = np.expand_dims(input_ops[key],axis=0)

    return input_ops

