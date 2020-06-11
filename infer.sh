#!/bin/bash

function vizdoom_full {
	CUDA_VISIBLE_DEVICES=$1 python solver.py --dataset_path $2 --dataset_type vizdoom --num_k 25 --checkpoint $3 --filtering $4
}

function vizdoom_if_else {
	CUDA_VISIBLE_DEVICES=$1 python solver.py --dataset_path $2 --dataset_type vizdoom --num_k 25 --checkpoint $3 --filtering $4
}

function karel {
	CUDA_VISIBLE_DEVICES=$1 python solver.py --dataset_path $2 --dataset_type karel --checkpoint $3 --filtering $4
}

function generalization {
	CUDA_VISIBLE_DEVICES=$1 python solver.py --dataset_path $2 --dataset_type vizdoom --checkpoint $3 --filtering dynamic --num_k $4
}

case "$1" in
	"karel")
		echo "Karel experiment"
		karel $2 $3 $4 $5;;
	"vizdoom_if_else")
		echo "Vizdoom if-else experiment"
		vizdoom_if_else $2 $3 $4 $5;;
	"vizdoom")
		echo "Vizdoom full experiment"
		vizdoom_full $2 $3 $4 $5;;
	"generalization")
		echo "Generalization experiment"
		generalization $2 $3 $4 1
		generalization $2 $3 $4 5
		generalization $2 $3 $4 10
		generalization $2 $3 $4 15
		generalization $2 $3 $4 20
		generalization $2 $3 $4 30
		generalization $2 $3 $4 35
		generalization $2 $3 $4 40;;
	"*")
		echo "No such experiment.";;
esac
