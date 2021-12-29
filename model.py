import torch
import torch.nn as nn
import torch.nn.functional as fc


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, batch_first, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=False,
                            bidirectional=False, num_layers=1)
        self.init_params()
        self.dropout = nn.Dropout(p=0.2)

    def init_params(self):
        for i in range(self.lstm.num_layers):
            nn.init.orthogonal_(getattr(self.lstm, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.lstm, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.lstm, f'bias_hh_l{i}'))
            nn.init.constant_(getattr(self.lstm, f'bias_ih_l{i}'))

    def forward(self, x):
        return self.init_params()


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.kaiming_normal_(self.linear.bias)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class BiDaf(nn.Module):
    def __init__(self, data, args):
        super(BiDaf, self).__init__()
        nn.modules.rnn.LSTM()

        # %% character embedding
        self.c_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)  # before init?
        self.nn.init.uniform(self.c_matrix.weight, -0.01, 0.01)
        self.CharConv = nn.Sequential(nn.Conv2d(1, args.chan_size, (args.char_dim, args.char_width)), nn.ReLU())
        # print(c_matrix.weight)

        # %% word embedding
        self.w_emb = nn.Embedding.from_pretrained(data.word.vocab.vectors, freeze=True)

        # %% highway
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2), nn.Sigmoid()))
        # %% contextual embedding
        self.contextual_lest = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True,
                                    batch_first=True, dropout=args.dropout)
        # %% attention flow
        self.att_c = Linear(args.hidden_size * 2, 1)
        self.att_q = Linear(args.hidden_size * 2, 1)
        self.att_cq = Linear(args.hidden_size * 2, 1)
        # %% modeling
        self.modeling_lstm = LSTM(input_size=args.hidden_size * 8, hidden_size=args, bidirectional=True,
                                  batch_first=True, dropout=args.dropout)

        # %% output

        # %%
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, batch):
        def char_embedding_layer(x):
            """
            in : (batch, seq_len, word_len)
            out: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            seq_len = x.size(1)

            x = self.dropout(self.c_emb(x))
            # (batch, seq_len, word_len, char_dim)

            x = x.transpose(2, 3)
            # (batch, seq_len, char_dim, word_len)

            x = x.view(-1, x.size(2), x.size(3)).unsqueeze(1)
            x = self.CharConv(x).squeeze()
            # (batch*seq_len, char_channel_size, (word_len-kernel_len+1))

            x = fc.max_pool1d(x, x.size(2)).squeeze()
            # (batch*seq_len, char_channel_size)

            x = x.view(batch_size, seq_len, -1)
            # (batch, seq_len, char_channel_size)
            return x

        def highway(x1, x2):
            x = torch.cat([x1, x2], dim=-1)
            # make clear the concept of highway network tomorrow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # %% character embedding
        c_char = char_embedding_layer(batch.c_char)
        q_char = char_embedding_layer(batch.q_char)

        # %% word embedding
        c_word = self.w_emb(batch.c_word[0])
        c_word_len = batch.c_word[1]

        q_word = self.w_emb(batch.q_word[0])
        q_word_len = batch.q_word[1]

        # %% highway
