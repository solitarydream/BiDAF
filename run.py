import argparse as ap

import torch
import torch as th

import dataloader


def train(args, data):
    #model = BiDaf()
    for i, batch in enumerate(data.train_iterator):
        print(batch)

    return 0


def run():
    parser = ap.ArgumentParser()
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--train-batch-size', default=3, type=int)
    parser.add_argument('--dev-batch-size', default=2, type=int)
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    #
    args = parser.parse_args()
    data = dataloader.dataloader(args)
    for i, batch in enumerate(data.tr_itr):
        if i == 1:
            print(batch)
    best_model = train(args, data)
    a = best_model


if __name__ == '__main__':
    run()
