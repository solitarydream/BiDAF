import torch
import torch.nn as nn
import torch.nn.functional as fc


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, bidirectional=bidirectional)
        self.init_params()
        self.dropout = nn.Dropout(p=dropout)

    def init_params(self):
        for i in range(self.lstm.num_layers):
            nn.init.orthogonal_(getattr(self.lstm, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.lstm, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.lstm, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.lstm, f'bias_ih_l{i}'), val=0)
            attrs = getattr(self.lstm, f'bias_hh_l{i}')
            #print(attrs)
            length = len(attrs)
            # print(length)
            secs = attrs.split(int(length / 4))
            # print(secs[0].shape,secs[1].shape,secs[2].shape,secs[3].shape,len(secs))
            # print(list(secs))
            # sec_list =[x for x in (secs)]
            sec_list = []
            for k in range(4):
                sec_list.append(secs[k])
            sec_list[1] = torch.ones(sec_list[1].shape)
            final = torch.cat(sec_list, dim=0)
            # print(type(final),type(attrs))
            setattr(self.lstm, f'bias_hh_l{i}', nn.parameter.Parameter(final))

            if self.lstm.bidirectional:
                nn.init.orthogonal_(getattr(self.lstm, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.lstm, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.lstm, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.lstm, f'bias_ih_l{i}_reverse'), val=0)
                attrs = getattr(self.lstm, f'bias_hh_l{i}_reverse')
                length = len(attrs)
                secs = attrs.split(int(length / 4))
                # print(secs[0].shape,secs[1].shape,secs[2].shape,secs[3].shape,len(secs))
                # print(list(secs))
                # sec_list =[x for x in (secs)]
                sec_list = []
                for k in range(4):
                    sec_list.append(secs[k])
                sec_list[1] = torch.ones(sec_list[1].shape)
                final = torch.cat(sec_list, dim=0)
                setattr(self.lstm, f'bias_hh_l{i}_reverse', nn.parameter.Parameter(final))
                # getattr(self.lstm, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)
                # this in-place operation is no more available

    def forward(self, x):
        # [(batch, seq_len, 2*char_channel_size), seq_len list] where seq_len represents actual length before padding
        x, len_list = x

        len_sorted, new_idx = torch.sort(len_list, descending=True)
        x_sorted = x.index_select(dim=0, index=new_idx)
        _, ori_index = torch.sort(new_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, len_sorted, batch_first=True)
        x_packed, (h, c) = self.lstm(x_packed)

        x_unpacked = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=ori_index)
        # (batch, seq_len, 2*char_channel_size)

        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=ori_index)
        # (batch, 2*char_channel_size)
        return x, h


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class BiDaf(nn.Module):
    def __init__(self, args=None, word_vectors=None):
        super(BiDaf, self).__init__()
        # self.args = args
        # %% character embedding
        self.c_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)  # before init?
        nn.init.uniform_(self.c_emb.weight, -0.01, 0.01)
        self.CharConv = nn.Sequential(nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
                                      nn.ReLU())
        # print(c_matrix.weight)

        # %% word embedding
        self.w_emb = nn.Embedding.from_pretrained(word_vectors, freeze=True)

        # %% highway
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2), nn.Sigmoid()))
        # %% contextual embedding
        self.contextual_lstm = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True,
                                    batch_first=True, dropout=args.dropout)
        # %% affine before calculating attention
        self.att_c = Linear(args.hidden_size * 2, 1)
        self.att_q = Linear(args.hidden_size * 2, 1)
        self.att_cq = Linear(args.hidden_size * 2, 1)
        # %% modeling
        self.modeling_lstm1 = LSTM(input_size=args.hidden_size * 8, hidden_size=args.hidden_size, bidirectional=True,
                                   batch_first=True, dropout=args.dropout)
        self.modeling_lstm2 = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True,
                                   batch_first=True, dropout=args.dropout)

        # %% output
        self.p1_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.p1_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.p2_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.p2_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.output_lstm = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True,
                                batch_first=True, dropout=args.dropout)

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
            for i in range(2):
                c = getattr(self, f'highway_gate{i}')(x)
                h = getattr(self, f'highway_linear{i}')(x)
                x = c * h + (1 - c) * x
            # (batch, seq_len, 2*char_channel_size)
            return x

        def att_flow(c, q):
            """
            :param c: (batch, c_seq_len, 2*char_channel_size)
            :param q: (batch, q_seq_len, 2*char_channel_size)
            :return:  (batch, c_seq_len, q_seq_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)
            cq = []
            for i in range(q_len):
                qi = q.select(1, i).unsqueeze(1)
                # (batch, 1, 2*char_channel_size)
                ci = self.att_weight_cq(h * qi).squeeze()
                # (batch, c_seq_len)
                cq.append(ci)
            # (q_len, batch, c_len)
            cq = torch.stack(cq, dim=-1)
            # ( batch, c_len, q_len)

            att_matrix = self.att_weight_c(c).expand(-1, -1, q_len) + \
                         self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + cq
            # (batch, c_len, q_len)

            a = fc.softmax(att_matrix, dim=2)  # not a view
            c2q_att = torch.bmm(a, q)
            # (batch, c_len, 2*char_channel_size)

            b = fc.softmax(torch.max(att_matrix, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len)
            q2c_att = torch.bmm(b, c).squeeze()
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # (batch, c_len, 2*char_channel_size)

            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            # (batch, c_len, 8*char_channel_size)

            return x

        def output(g, m, seq_len):
            """
                :param g: (batch, seq_len, 8*char_channel_size)
                :param m: (batch, seq_len , 2*char_channel_size)
                :return: p1: (batch, seq_len), p2: (batch, c_len)
            """
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()

            m2 = self.output_LSTM((m, l))[0]  # why lstm again?????????????????????
            # (batch, c_len, 2*char_channel_size)

            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            # (batch, seq_len)
            return p1, p2

        # %% character embedding
        c_char = char_embedding_layer(batch.c_char)
        q_char = char_embedding_layer(batch.q_char)

        # %% word embedding
        c_word = self.w_emb(batch.c_word[0])
        c_word_len = batch.c_word[1]

        q_word = self.w_emb(batch.q_word[0])
        q_word_len = batch.q_word[1]

        # %% highway
        c = highway(c_char, c_word)
        q = highway(q_char, q_word)
        # (batch, seq_len, 2*char_channel_size)

        # %% contextual embedding
        c = self.contextual_lstm((c, c_word_len))[0]
        q = self.contextual_lstm((q, q_word_len))[0]
        # x: (batch, seq_len, 2*char_channel_size)
        # h: (batch, 2*char_channel_size) => dropped

        # %% attention flow
        g = att_flow(c, q)

        # %% modeling layer
        m = self.modeling_lstm1((g, c_word_len))[0]
        m = self.modeling_lstm2((m, c_word_len))[0]
        # (batch, c_seq_len, 2 * char_channel_size)

        # %% output layer
        p1, p2 = output(g, m, c_word_len)
