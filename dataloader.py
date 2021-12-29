import pprint
import torch
from torch.utils.data import DataLoader
import json
import os
import nltk
import re
from torchtext.legacy import data
from torchtext.vocab import GloVe


# %%
class Squad():
    def __init__(self):
        self.train = None
        self.dev = None
        self.raw = None
        self.char = None
        self.word = None
        self.label = None
        self.train = None
        self.train = None
        self.tr_itr = None
        self.dev_itr = None


def process_abnormal(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


# %%
def preprocess(in_path='data/squad/dev-v1.1.json'):
    out_path = in_path + 'l'
    if os.path.exists(out_path):
        return None
    dump = []
    with open(in_path, 'r', encoding='utf-8') as resource_file:
        raw_data = json.load(resource_file)['data']
        for nar, article in enumerate(raw_data):
            for npa, para in enumerate(article['paragraphs']):
                con = para['context'].replace("''", '"').replace("``", '"')
                w_tokens = list(map(nltk.word_tokenize, nltk.sent_tokenize(con)))
                w_tokens = [process_abnormal(tokens) for tokens in w_tokens]
                c_tokens = [[list(word) for word in sent] for sent in w_tokens]
                # if nar==npa==1:
                # print(w_tokens)
                for nqa, pair in enumerate(para['qas']):
                    qs = pair['question']
                    # print(qs)
                    qas_id = pair['id']
                    for nans, ans in enumerate(pair['answers']):
                        answer = ans['text']
                        cha_s_id = ans['answer_start']
                        cha_e_id = cha_s_id + len(answer)
                        dump.append(dict(
                            [('id', qas_id), ('context', con), ('question', qs), ('answer', answer), ('s_id', cha_s_id),
                             ('e_id', cha_e_id)]))

    if not os.path.exists(out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)


# %%
def dataloader(args, dir_path='./data/squad/'):
    path = dir_path + 'tokenized/'
    train_in_path = dir_path + 'dev-v1.1.jsonl'
    dev_in_path = dir_path + 'dev-v1.1.jsonl'
    train_out_path = path + 'train.pt'
    dev_out_path = path + 'dev.pt'

    if not os.path.exists(train_in_path):
        preprocess(dir_path + 'dev-v1.1.json')
    if not os.path.exists(dev_in_path):
        preprocess(dir_path + 'dev-v1.1.json')

    raw = data.RawField()
    char_nesting = data.Field(batch_first=True, tokenize=list, lower=True)
    char = data.NestedField(char_nesting, tokenize=nltk.word_tokenize)
    word = data.Field(batch_first=True, tokenize=nltk.word_tokenize, lower=True, include_lengths=True)
    label = data.Field(sequential=False, unk_token=None, use_vocab=False)
    dict_fields = {'id': ('id', raw), 's_id': ('s_idx', label), 'e_id': ('e_idx', label),
                   'context': [('c_word', word), ('c_char', char)], 'question': [('q_word', word), ('q_char', char)]}

    list_fields = [('id', raw), ('s_idx', label), ('e_idx', label),
                   ('c_word', word), ('c_char', char),
                   ('q_word', word), ('q_char', char)]
    # %%
    if not os.path.exists(dir_path + 'tokenized/'):
        train, dev = data.TabularDataset.splits(
            path='./data/squad/',
            train='dev-v1.1.jsonl',
            validation='dev-v1.1.jsonl',
            format='json',
            fields=dict_fields)
        os.mkdir(path)
        torch.save(train.examples, train_out_path)
        torch.save(dev.examples, dev_out_path)
    else:
        train_examples = torch.load(train_out_path)
        dev_examples = torch.load(dev_out_path)
        train = data.Dataset(examples=train_examples, fields=list_fields)
        dev = data.Dataset(examples=dev_examples, fields=list_fields)

    char.build_vocab(train, dev)
    word.build_vocab(train, dev, vectors=GloVe(name='6B', dim=args.word_dim))
    # %%
    train_iterator = data.BucketIterator(train, batch_size=args.train_batch_size, sort_key=lambda x: x.c_word,
                                         device='cuda', repeat=True, shuffle=True)
    dev_iterator = data.BucketIterator(dev, batch_size=args.dev_batch_size, sort_key=lambda x: x.c_word,
                                       device='cuda', repeat=True, shuffle=True)

    dt = Squad()
    dt.train = train
    dt.dev = train
    dt.raw = raw
    dt.char = char
    dt.word = word
    dt.label = label
    dt.tr_itr = train_iterator
    dt.dev_itr = dev_iterator
    return dt


if __name__ == '__main__':
    dataloader()
