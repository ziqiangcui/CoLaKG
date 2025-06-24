#!/bin/bash

decay=1e-4
lr=0.001
layer=3
seed=2020
dataset="ml-1m"
topks="[10,20]"
recdim=64
use_drop_edge=0
keepprob=1.0
batch_size=4096
dropout_i=0.6
dropout_u=0.6
dropout_n=0.6
neighbor_k=30
item_semantic_emb_file='../data/ml-1m/movie_embeddings_simcse_kg.pt'
user_semantic_emb_file='../data/ml-1m/movie_embeddings_simcse_kg_user.pt'

CUDA_VISIBLE_DEVICES=0 python main.py --bpr_batch=$batch_size --decay=$decay --lr=$lr --layer=$layer --seed=$seed --dataset=$dataset --topks=$topks --recdim=$recdim --use_drop_edge=$use_drop_edge --keepprob=$keepprob --neighbor_k=$neighbor_k --dropout_i=$dropout_i --dropout_u=$dropout_u --dropout_n=$dropout_n --item_semantic_emb_file=$item_semantic_emb_file --user_semantic_emb_file=$user_semantic_emb_file