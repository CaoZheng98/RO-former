import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_inputs, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_outputs = enc_inputs.clone()
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_outputs, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).any(dim=-1, keepdim=False).unsqueeze(-2) if (seq.dim()>2) else (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size() if seq.ndim == 2 else [seq.size()[0], seq.size()[1]]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v,d_model, d_inner, n_position=2000, dropout=0.1, scale_emb=False):

        super().__init__()

        self.position_enc = PositionalEncoding(d_hid=d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []
        dec_output = trg_seq.clone()
        # -- Forward

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class Transformer(nn.Module):

    def __init__(self, src_len_max, trg_len_max, src_pad_idx, trg_pad_idx, d_src, d_trg, d_model, trg_size, scale_emb=False, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_pos_enc = PositionalEncoding(d_hid=d_model, n_position=src_len_max)
        self.trg_pos_enc = PositionalEncoding(d_hid=d_model, n_position=trg_len_max)
        self.src_linear = nn.Linear(d_src, d_model)
        self.trg_linear = nn.Linear(d_trg, d_model)
        self.scale_emb = scale_emb
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(n_layers=2, n_head=4, d_k=16, d_v=16, d_model=64, d_inner=256)
        self.decoder = Decoder(n_layers=2, n_head=4, d_k=16, d_v=16,d_model=64, d_inner=256)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear_project = nn.Linear(d_model, trg_size)
    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        if src_seq.ndim == 2:
            src_seq = src_seq.unsqueeze(-1)
        else:
            src_seq = src_seq
        src_inputs = self.src_linear(src_seq)
        if self.scale_emb:
            src_inputs *= self.d_model ** 0.5
        src_inputs = self.dropout(self.src_pos_enc(src_inputs))
        src_inputs = self.layer_norm(src_inputs)
        enc_output = self.encoder(src_inputs, src_mask)
        if trg_seq.ndim == 2:
            trg_inputs = trg_seq.unsqueeze(-1)
        else:
            trg_inputs = trg_seq
        trg_inputs = self.trg_linear(trg_inputs)
        if self.scale_emb:
            trg_inputs *= self.d_model ** 0.5
        trg_inputs = self.dropout(self.trg_pos_enc(trg_inputs))
        trg_inputs = self.layer_norm(trg_inputs)
        dec_output, *_ = self.decoder(trg_inputs, trg_mask, enc_output, src_mask)
        # Linear project
        project_outputs = self.linear_project(dec_output)

        return project_outputs
#
#     class Transformer(nn.Module):
#
#         def __init__(self, src_len_max, trg_len_max, src_pad_idx, trg_pad_idx, d_src, d_trg, d_model, trg_size,
#                      scale_emb=False, dropout=0.1):
#             super(Transformer, self).__init__()
#             self.src_pad_idx = src_pad_idx
#             self.trg_pad_idx = trg_pad_idx
#             self.src_pos_enc = PositionalEncoding(d_hid=d_model, n_position=src_len_max)
#             self.trg_pos_enc = PositionalEncoding(d_hid=d_model, n_position=trg_len_max)
#             self.src_linear = nn.Linear(d_src, d_model)
#             self.trg_linear = nn.Linear(d_trg, d_model)
#             self.scale_emb = scale_emb
#             self.dropout = nn.Dropout(p=dropout)
#             self.encoder = Encoder(n_layers=2, n_head=4, d_k=16, d_v=16, d_model=64, d_inner=256)
#             self.decoder = Decoder(n_layers=2, n_head=4, d_k=16, d_v=16, d_model=64, d_inner=256)
#
#             # 新增GRU层和线性投影
#             self.gru = nn.GRU(
#                 input_size=d_model,
#                 hidden_size=d_model,
#                 num_layers=1,
#                 batch_first=True,
#                 dropout=dropout if 1 > 1 else 0  # 如果是多层使用dropout
#             )
#             self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#             self.linear_project = nn.Linear(d_model, trg_size)
#
#         def forward(self, src_seq, trg_seq):
#             src_mask = get_pad_mask(src_seq, self.src_pad_idx)
#             trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
#
#             # 源序列处理
#             if src_seq.ndim == 2:
#                 src_seq = src_seq.unsqueeze(-1)
#             src_inputs = self.src_linear(src_seq)
#             if self.scale_emb:
#                 src_inputs *= self.d_model ** 0.5
#             src_inputs = self.dropout(self.src_pos_enc(src_inputs))
#             src_inputs = self.layer_norm(src_inputs)
#             enc_output = self.encoder(src_inputs, src_mask)
#
#             # 目标序列处理
#             if trg_seq.ndim == 2:
#                 trg_inputs = trg_seq.unsqueeze(-1)
#             else:
#                 trg_inputs = trg_seq
#             trg_inputs = self.trg_linear(trg_inputs)
#             if self.scale_emb:
#                 trg_inputs *= self.d_model ** 0.5
#             trg_inputs = self.dropout(self.trg_pos_enc(trg_inputs))
#             trg_inputs = self.layer_norm(trg_inputs)
#
#             # Decoder输出
#             dec_output, *_ = self.decoder(trg_inputs, trg_mask, enc_output, src_mask)
#
#             # 新增GRU处理
#             dec_output, _ = self.gru(dec_output)  # (batch_size, seq_len, d_model)
#
#             # 线性投影
#             project_outputs = self.linear_project(dec_output)
#
#             return project_outputs


# class Transformer(nn.Module):
#
#     def __init__(self, src_len_max, trg_len_max, src_pad_idx, trg_pad_idx, d_src, d_trg, d_model, trg_size,
#                  scale_emb=False, dropout=0.1):
#         super(Transformer, self).__init__()
#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx
#         self.src_pos_enc = PositionalEncoding(d_hid=d_model, n_position=src_len_max)
#         self.trg_pos_enc = PositionalEncoding(d_hid=d_model, n_position=trg_len_max)
#         self.src_linear = nn.Linear(d_src, d_model)
#         self.trg_linear = nn.Linear(d_trg, d_model)
#         self.scale_emb = scale_emb
#         self.dropout = nn.Dropout(p=dropout)
#         self.encoder = Encoder(n_layers=2, n_head=4, d_k=16, d_v=16, d_model=64, d_inner=256)
#         self.decoder = Decoder(n_layers=2, n_head=4, d_k=16, d_v=16, d_model=64, d_inner=256)
#
#         # 新增GRU层和线性投影
#         self.gru = nn.GRU(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True,
#             dropout=dropout if 1 > 1 else 0  # 如果是多层使用dropout
#         )
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         self.linear_project = nn.Linear(d_model, trg_size)

        # # 示例：更复杂的LSTM配置
        # self.lstm = nn.LSTM(
        #     input_size=d_model,
        #     hidden_size=d_model * 2,  # 扩大隐藏层
        #     num_layers=2,  # 加深层数
        #     bidirectional=True,  # 双向结构
        #     dropout=0.2 if 2 > 1 else 0,
        #     batch_first=True
        # )
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # 需要相应调整后续的线性层输入维度
        # self.linear_project = nn.Linear(d_model*4, trg_size)
    # def forward(self, src_seq, trg_seq):
    #     src_mask = get_pad_mask(src_seq, self.src_pad_idx)
    #     trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
    #
    #     # 源序列处理
    #     if src_seq.ndim == 2:
    #         src_seq = src_seq.unsqueeze(-1)
    #     src_inputs = self.src_linear(src_seq)
    #     if self.scale_emb:
    #         src_inputs *= self.d_model ** 0.5
    #     src_inputs = self.dropout(self.src_pos_enc(src_inputs))
    #     src_inputs = self.layer_norm(src_inputs)
    #     enc_output = self.encoder(src_inputs, src_mask)
    #
    #     # 目标序列处理
    #     if trg_seq.ndim == 2:
    #         trg_inputs = trg_seq.unsqueeze(-1)
    #     else:
    #         trg_inputs = trg_seq
    #     trg_inputs = self.trg_linear(trg_inputs)
    #     if self.scale_emb:
    #         trg_inputs *= self.d_model ** 0.5
    #     trg_inputs = self.dropout(self.trg_pos_enc(trg_inputs))
    #     trg_inputs = self.layer_norm(trg_inputs)
    #
    #     # Decoder输出
    #     dec_output, *_ = self.decoder(trg_inputs, trg_mask, enc_output, src_mask)
    #
    #     # 新增GRU处理
    #     # dec_output, _ = self.lstm(dec_output)  # (batch_size, seq_len, d_model)
    #     dec_output, _ = self.gru(dec_output)
    #     # 线性投影
    #     project_outputs = self.linear_project(dec_output)
    #
    #     return project_outputs
