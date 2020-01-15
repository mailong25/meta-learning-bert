import json
from random import shuffle
from collections import Counter
import torch
from transformers import BertModel, BertTokenizer
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from reptile import Learner
from task import MetaTask
import random
import numpy as np

def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", default='dataset.json', type=str,
                        help="Path to dataset file")
    
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Path to bert model")
    
    parser.add_argument("--num_labels", default=2, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=5, type=int,
                        help="Number of outer interation")
    
    parser.add_argument("--k_spt", default=80, type=int,
                        help="Number of support samples per task")
    
    parser.add_argument("--k_qry", default=20, type=int,
                        help="Number of query samples per task")

    parser.add_argument("--outer_batch_size", default=2, type=int,
                        help="Batch of task size")
    
    parser.add_argument("--inner_batch_size", default=12, type=int,
                        help="Training batch size in inner iteration")
    
    parser.add_argument("--outer_update_lr", default=5e-5, type=float,
                        help="Meta learning rate")
    
    parser.add_argument("--inner_update_lr", default=5e-5, type=float,
                        help="Inner update learning rate")
    
    parser.add_argument("--inner_update_step", default=10, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--inner_update_step_eval", default=40, type=int,
                        help="Number of interation in the inner loop during test time")
    
    parser.add_argument("--num_task_train", default=500, type=int,
                        help="Total number of meta tasks for training")
    
    parser.add_argument("--num_task_test", default=3, type=int,
                        help="Total number of tasks for testing")
    
    args = parser.parse_args()
    
    reviews = json.load(open(args.data))
    low_resource_domains = ["office_products", "automotive", "computer_&_video_games"]

    train_examples = [r for r in reviews if r['domain'] not in low_resource_domains]
    test_examples = [r for r in reviews if r['domain'] in low_resource_domains]
    print(len(train_examples), len(test_examples))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = True)
    learner = Learner(args)
    
    test = MetaTask(test_examples, num_task = args.num_task_test, k_support=args.k_spt, 
                    k_query=args.k_qry, tokenizer = tokenizer)

    global_step = 0
    for epoch in range(args.epoch):

        train = MetaTask(train_examples, num_task = args.num_task_train, k_support=args.k_spt, 
                         k_query=args.k_qry, tokenizer = tokenizer)

        db = create_batch_of_tasks(train, is_shuffle = True, batch_size = args.outer_batch_size)

        for step, task_batch in enumerate(db):

            acc = learner(task_batch)

            print('Step:', step, '\ttraining Acc:', acc)

            if global_step % 20 == 0:
                random_seed(123)
                print("\n-----------------Testing Mode-----------------\n")
                db_test = create_batch_of_tasks(test, is_shuffle = False, batch_size = 1)
                acc_all_test = []

                for test_batch in db_test:
                    acc = learner(test_batch, training = False)
                    acc_all_test.append(acc)

                print('Step:', step, 'Test F1:', np.mean(acc_all_test))

                random_seed(int(time.time() % 10))

            global_step += 1
            
if __name__ == "__main__":
    main()