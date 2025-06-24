
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go Model")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--neighbor_k', type=int,default=10,
                        help="the num of neighbors")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--use_drop_edge', type=int,default=1,
                        help="using the drop_edge or not for lightgcn")
    parser.add_argument('--keepprob', type=float,default=0.7,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--dropout_i', type=float,default=0.6,
                        help="the dropout for item semantic embeddings")
    parser.add_argument('--dropout_u', type=float,default=0.6,
                        help="the dropout for user semantic embeddings")
    parser.add_argument('--dropout_n', type=float,default=0.6,
                        help="the dropout for neighbor embeddings")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='ml-1m',
                        help="available datasets")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--item_semantic_emb_file', type=str,default=" ",
                        help="the path of item_semantic_emb_file")
    parser.add_argument('--user_semantic_emb_file', type=str,default=" ",
                        help="the path of user_semantic_emb_file")
    parser.add_argument('--topks', nargs='?',default="[10,20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='colakg', help='rec-model, support [mf, lgn, colakg]')
    return parser.parse_args()
