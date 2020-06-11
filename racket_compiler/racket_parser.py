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

import ply.yacc as yacc
import ply.lex as lex


class Parser():

    def __init__(self, dataset_type):

        if dataset_type == "karel":
            actions = ['move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker']
            perceptions = ['frontIsClear',
                           'leftIsClear',
                           'rightIsClear',
                           'markersPresent',
                           'noMarkersPresent']
        else:
            actions = ['MOVE_FORWARD',
                       'MOVE_BACKWARD',
                       'MOVE_LEFT',
                       'MOVE_RIGHT',
                       'TURN_LEFT',
                       'TURN_RIGHT',
                       'ATTACK',
                       'SELECT_WEAPON1',
                       'SELECT_WEAPON3',
                       'SELECT_WEAPON4',
                       'SELECT_WEAPON5'
                       ]
            perceptions = ['ISTHERE Demon',
                           'ISTHERE HellKnight',
                           'ISTHERE Revenant',
                           'INTARGET Demon',
                           'INTARGET HellKnight',
                           'INTARGET Revenant']


        reserved = {
           'define' : 'DEFINE',
            'let' : 'LET',
            'if' : 'IF',
            'append': 'APPEND',
            'list': 'LIST',
            'loop': 'LOOP',
            'len': 'LEN',
            'Action_Block':'AB'
        }

        tokens = ['LPAR','RPAR','NAME','NUMBER','GEQ','MINUS'] + list(reserved.values())

        # Tokens

        t_LPAR = r'\('
        t_RPAR = r'\)'
        t_GEQ = r'>='
        t_MINUS = r'-'

        def t_NAME(t):
            r'[a-zA-Z_][a-zA-Z_0-9_-]*'
            t.type = reserved.get(t.value,'NAME')    # Check for reserved words
            return t

        def t_NUMBER(t):
            r'\d+'
            t.value = int(t.value)
            return t

        def t_newline(t):
            r'\n+'
            t.lexer.lineno += len(t.value)

        # Ignored characters
        t_ignore = " \t"

        def t_error(t):
            print("Illegal character !")
            t.lexer.skip(1)

        self.racket_lexer = lex.lex()

        def p_program(p):

            ''' program : LPAR DEFINE expression expression RPAR'''

            p[0] = "DEF run m( " + p[4] + " m)"


        def p_loop_1(p):
            ''' expression : LET LOOP expression LPAR IF LPAR NAME expression expression RPAR loop expression RPAR '''

            p[0] = "WHILE c( " + p[8] +" c) w( "+ p[11] + " w)"

        def p_loop_2(p):

            ''' loop : LPAR LOOP expression expression RPAR'''

            p[0] = p[3]

        def p_minus(p):

            ''' expression : MINUS LEN NUMBER'''

        def p_geq(p):

            ''' expression : GEQ LEN NUMBER'''

        def p_let(p):

            ''' expression : LET LPAR LPAR expression expression RPAR RPAR expression'''

            if p[5] == "":
                p[0] = p[8]
            elif p[8] == "":
                p[0] = p[5]
            else:
                p[0] = p[5] + " " + p[8]



        def p_len(p):

            ''' expression : LEN NUMBER'''
            p[0] = ""

        def p_list(p):

            ''' expression : LIST NUMBER'''
            p[0] = actions[p[2]]

        def p_list_empty(p):
            ''' expression : LIST'''
            p[0] = ""

        def p_append(p):

            ''' expression : APPEND expression expression'''

            if p[2] == "":
                p[0] = p[3]
            elif p[3] == "":
                p[0] = p[2]
            else:
                p[0] = p[2] + " " + p[3]
        def p_expression_par(p):

            '''expression : LPAR expression RPAR '''


            p[0] = p[2]

        def p_expression_name(p):

            '''expression : NAME '''

            p[0] = ""


        def p_expression_apply(p):

            '''expression : expression expression  '''

            if p[1] == "":
                p[0] = p[2]
            elif p[2] == "":
                p[0] = p[1]
            else:
                p[0] = p[1] + " " + p[2]

        def p_if(p):

            ''' expression : IF expression expression expression '''

            if p[4] == "":
                p[0] = "IF c( " + p[2] +" c) i( "+ p[3] + " i)"
            elif p[3] == "":
                p[0] = "IF c( not c( " + p[2] +" c) c) i( "+ p[4] + " i)"
            else:
                p[0] = "IFELSE c( " + p[2] +" c) i( "+ p[3] + " i) ELSE e( " + p[4] + " e)"

        def p_cond_exception(p):
            ''' expression : AB expression NUMBER '''

            p[0] = ""

        def p_cond(p):

            ''' expression : expression expression NUMBER '''
            p[0] = perceptions[p[3]]


        def p_cond_2(p):
            ''' expression : expression expression expression '''

            p[0] = ""



        self.racket_parser = yacc.yacc()



