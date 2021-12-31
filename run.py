import argparse as ap

import torch
import torch as th
from model import BiDaf
from torch import nn, optim

import dataloader


def train(data, args):
    device = torch.device('cuda:0')

    model = BiDaf(args, data.word.vocab.vectors).to(device)

    # for name, param in model.named_parameters():
    #     print(name)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)

    while True:
        pass
    # print(f'{name}: {param}' if param.requires_grad() else '')

    # for i, batch in enumerate(data.train_iterator):
    # print(batch)

    return 0


def run():
    parser = ap.ArgumentParser()
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--train-batch-size', default=3, type=int)
    parser.add_argument('--dev-batch-size', default=2, type=int)

    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)

    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-dim', default=8, type=int)

    parser.add_argument('--dropout', default=0.2, type=float)

    parser.add_argument('--learning-rate', default=0.5, type=float)
    #
    args = parser.parse_args()
    data = dataloader.dataloader(args)

    setattr(args, 'char_vocab_size', len(data.char.vocab))
    setattr(args, 'word_vocab_size', len(data.word.vocab))
    # for i, batch in enumerate(data.tr_itr):
    best_model = train(data, args)

    a = best_model


if __name__ == '__main__':
    run()

