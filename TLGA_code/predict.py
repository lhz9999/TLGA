

import jieba
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import data, config
from config import USE_CUDA, DEVICE
from data import Vocab
from model import Model
from batcher import Example
from batcher import Batch
from batcher import get_input_from_batch
from numpy import random


def build_batch_by_article(article, vocab):
    words = jieba.cut(article)
    art_str = " ".join(words)
    example = Example(art_str, ["",], vocab)
    ex_list = [example for _ in range(config.beam_size)]
    batch  = Batch(ex_list, vocab, config.beam_size)
    return batch

"""
decode阶段使用 beam search 算法
"""
class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens = self.tokens + [token],
                            log_probs = self.log_probs + [log_prob],
                            state = state,
                            context = context,
                            coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path, vocab):
        self.vocab = vocab
        # 加载模型
        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, batch):
        best_summary = self.beam_search(batch)

        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self.vocab,
                                                (batch.art_oovs[0] if config.pointer_gen else None))

        ###Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        print("decode_words:", decoded_words)
        return "".join(decoded_words)


    def beam_search(self, batch):
        in_features = config.hidden_dim * config.beam_size
        out_features = config.hidden_dim * config.beam_size
        alpha = 0.2 # 激活斜率 (LeakyReLU)的激活斜率
        W = nn.Parameter(torch.zeros(size=(in_features, out_features))).cuda() #建立一个w权重，用于对特征数F进行线性变化
        nn.init.xavier_uniform_(W.data, gain=1.414).cuda() #对权重矩阵进行初始化
        aa = nn.Parameter(torch.zeros(size=(2*out_features, 1))).cuda() #计算函数α，输入是上一层两个输出的拼接，输出的是eij，a的size为(2*F',1)
        nn.init.xavier_uniform_(aa.data, gain=1.414).cuda() #对a进行初始化
        leakyrelu = nn.LeakyReLU(alpha).cuda()#激活层
        a = []
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        att_visual = []
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if USE_CUDA:
                y_t_1 = y_t_1.to(DEVICE)
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            h_decode, c_decode = s_t_1
            #a.append((h_decode.view(-1)).cpu().detach().numpy().tolist())
            a.append((h_decode.view(-1)).detach().tolist())
            # b.append((c_decode.view(-1)).detach().tolist())
#更新h
            gat_input = torch.tensor(a).cuda()
            # print(gat_input.size())
            h_gat = torch.mm(gat_input, W).cuda()#    [N,Feature]
            # print(h_gat.size())
            N = gat_input.size()[0]##获取当前的节点数量
            out_size = torch.zeros(N,config.beam_size,config.hidden_dim).cuda()
            #h.repeat(1,N)将h的每一行按列扩展N次，扩展后的size为(N,F'*N)
            #.view(N*N,-1)对扩展后的矩阵进行重新排列，size为(N*N,F')每N行表示的都是同一个节点的N次重复的特征表示。
            #h.repeat(N,1)对当前的所有行重复N次，每N行表示N个节点的特征表示
            #torch.cat对view之后和repeat(N,1)的特征进行拼接，每N行表示一个节点的N次特征重复，分别和其他节点做拼接。size为(N*N,2*F')
            #.view(N,-1,2*self.out_features)表示将矩阵整理为(N,N,2*F')的形式。第一维度的每一个表示一个节点，第二个维度表示上一个节点对应的其他的所有节点，第三个节点表示特征拼接
            a_input = torch.cat([h_gat.repeat(1, N).view(N * N, -1), h_gat.repeat(N, 1)], dim=1).view(N, -1, 2 * out_features).cuda()
            #每一行是一个隐状态与其他各个隐状态的相关性值
            e = leakyrelu(torch.matmul(a_input, aa).squeeze(2)).cuda()#[N,N]
            #做一个softmax，生成贡献度权重
            decoder_attention = F.softmax(e, dim=1).cuda()

            # att_numpy = decoder_attention.detach().cpu().numpy()
            # att.append(att_numpy[steps-1])
            # print(att_numpy[steps-1])
            # print(att_numpy[steps-1].tolist())
            # print(a_input.size())
            #根据权重计算最终的特征输出。

            lstm_out_attention = torch.matmul(decoder_attention, h_gat).cuda()
            lstm_out_attention = F.elu(lstm_out_attention).cuda()

            lstm_out_attention = lstm_out_attention.view_as(out_size).cuda()#[len,batch,dim]

            h_decoder = lstm_out_attention[-1,:,:].cuda()#更新decoder隐状态，

            h_decoder = h_decoder.unsqueeze(0).cuda()
            s_t_1 =  tuple([h_decoder,c_decode])#更新st值
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)

            att_numpy = attn_dist.detach().cpu().numpy()
            output_att = att_numpy[0]
            att_visual.append(output_att.tolist())



            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)
            # print(p_gen)
            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1
        final_attenion = np.array(att_visual)
        print(final_attenion)
        np.save('/data1/lhz/pg_network_torch/transformer_gat_fin/history_attenion.npy',final_attenion)
        print('The End')
        
        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    article = '据 中 国 经 营 报 报 道 继 裁 员 风 波 之 后 为 缓 解 业 绩 下 滑 的 困 局 比 亚 迪 又 采 取 了 全 员 降 薪 措 施 与 去 年 业 绩 下 滑 时 的 裁 员 对 象 多 为 普 通 员 工 不 同 比 亚 迪 此 次 降 薪 举 动 的 范 围 包 括 了 管 理 层 比 亚 迪 一 位 内 部 员 工 表 示 降 薪 十 分 影 响 工 作 积 极 性 '
    # article ='中 石 化 董 事 长 王 玉 普 与 俄 罗 斯 石 油 公 司 总 裁 谢 钦 签 订 了 共 同 开 发 鲁 斯 科 耶 油 气 田 和 尤 鲁 勃 切 诺 托 霍 姆 油 气 田 合 作 框 架 协 议 根 据 协 议 中 石 化 集 团 有 权 收 购 俄 罗 斯 石 油 公 司 所 属 东 西 伯 利 亚 油 气 公 司 和 秋 明 油 气 公 司 这 两 家 公 司 4 9 的 股 份 ' 
    model_path = sys.argv[1]
    
    vocab = Vocab(config.vocab_path, config.vocab_size)
    batch = build_batch_by_article(article, vocab)
    beam_processor = BeamSearch(model_path, vocab)
    beam_processor.decode(batch)