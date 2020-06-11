#!/bin/bash

function vizdoom {
	CUDA_VISIBLE_DEVICES=$1 python trainer.py --dataset_path $2 --dataset_type vizdoom --num_k 25 --batch_size 8 --output $3 --log_step 1000 --write_summary_step 1000 --test_sample_step 1000 --max_steps 50000
}

function vizdoom_if_else {
	CUDA_VISIBLE_DEVICES=$1 python trainer.py --dataset_path $2 --dataset_type vizdoom --num_k 25 --batch_size 8 --output $3 --log_step 200 --write_summary_step 200 --test_sample_step 200 --max_steps 10000

}

function karel {
	CUDA_VISIBLE_DEVICES=$1 python trainer.py --dataset_path $2 --dataset_type karel --output $3 --max_steps 10000
}

case "$1" in
	"karel")
		echo "Karel training"
		karel $2 $3 karel_1
		karel $2 $3 karel_2
		karel $2 $3 karel_3;;
	"vizdoom")
		echo "Vizdoom training"
		vizdoom $2 $3 vizdoom_1
		vizdoom $2 $3 vizdoom_2
		vizdoom $2 $3 vizdoom_3;;
	"vizdoom_if_else")
		echo "Vizdoom if-else training"
		vizdoom_if_else $2 $3 vizdoom_if_else_1
		vizdoom_if_else $2 $3 vizdoom_if_else_2
		vizdoom_if_else $2 $3 vizdoom_if_else_3;;
	"*")
		echo "No such experiment.";;
esac
