# PLANS: Neuro-Symbolic Program Learning from Videos

This is the source code for the Neurips 2020 paper **[PLANS: Neuro-Symbolic Program Learning from Videos](https://proceedings.neurips.cc/paper/2020/file/fe131d7f5a6b38b23cc967316c13dae2-Paper.pdf)**.

## Description

This work presents a new approach to program learning from raw, unstructured input. It combines a neural architecture that is trained to extract abstract, high-level specifications of program behaviour from high-dimensional sensory input, and a rule-based solver that provides correctness guarantees regarding the generated program.
We reach state-of-the-art performance on the task of *program synthesis from demonstration videos*. We address the crucial problem of making our system resistant to noise originating from the imperfect accuracy of the neural component.

![PLANS](image.png?raw=true "Model illustration.")

## Structure of the repository

- **models/** contains the implementation of the neural architecture for action and perception recognition.
- **rosette/** contains the syntax files that encode the DSL.
- **racket_compiler/** contains a *lex-yacc* implementation of a Racket to Python parser for the synthesized programs.
- **third_party/** contains the *demo2program* repository as a submodule.

## Installation


### Dependencies

#### Rosette solver

For installing Racket and the Rosette solver, please see the instructions in the [official repository](https://github.com/emina/rosette).

#### PLY

[Python Lex-Yacc](https://www.dabeaz.com/ply/) is an implementation of lex and yacc parsing tools for Python.

#### Demo2program

This project builds on code provided in the [demo2program](https://github.com/shaohua0116/demo2program) repository. Our repository contains it as a submodule, in **third_party/demo2program/**.

In order to be able to import code from this submodule, run

        git submodule init
        git submodule update    

to get the submodule. Then, execute

        touch third_party/demo2program/__init__.py

We have modified some of the files in order to
1. solve an issue in the pre-processing of demonstrations that led to incorrect sequence padding.
2. create a new input operation to suit our needs.

The modified files can be found in **third_party/modified/**. We will suggest these modifications in the original *demo2program* repository.

### Setting-up the virtual environment

Execute the following to set up the conda environment, and install the required python packages

        conda create -n rpl_env python=2.7
        conda activate rpl_env    
        conda install -c anaconda -c conda-forge -c pytorch --file requirements.txt

## Reproducing experiments

### Training neural models

The following command will train the neural action and perception recognition models.

         ./train.sh <dataset type> <GPU Id> <dataset path> 

- ``<dataset type>`` can be either *karel*, *vizdoom* or *vizdoom_if_else*.

- ``<dataset path>`` is the relative or absolute path to the corresponding dataset.
- Set ``<GPU Id>`` to -1 for CPU training.

The training logs and model checkpoints are saved in the **train_dir/** directory.

### Running the solver

The following command will attempt to solve all the test instances in the dataset with the Rosette solver.

         ./infer.sh <experiment> <GPU Id> <dataset path> <checkpoint path> <filtering>

- ``<experiment>`` can be either *karel*, *vizdoom*, *vizdoom_if_else*, or *generalization*.

- ``<dataset path>`` is the relative or absolute path to the corresponding dataset.
- ``<checkpoint path>`` is the relative or absolute path to the model checkpoint obtained in the previous step.

- ``<filtering>`` can be *none*, *static* or *dynamic*.

Resulting accuracy will be reported in the **solver_logs/** directory.

## Datasets, Trained models, Experimental Results

We evaluated our algorithm on the downloadable datasets provided with the [demo2program](https://github.com/shaohua0116/demo2program) repository.  
Our training logs and checkpoints are available on this [Google drive](https://drive.google.com/drive/folders/1CLbL4wSjYfvuMuwTh2aj91-i27FOa3_K?usp=sharing).

The following is a summary of the experimental results presented in the paper.

![Results](results_summary.png?raw=true "Summary of experimental results (from the associated paper).")

## Citing this work

If you use this repository, please cite the following work
'''
@article{dang2020plans,
  title={PLANS: Neuro-Symbolic Program Learning from Videos},
  author={Dang-Nhu, Rapha{\"e}l},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
'''

