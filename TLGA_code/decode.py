
"""
decode阶段使用 beam search 算法
"""
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import data
import config
from data import Vocab
from model import Model
from config import USE_CUDA, DEVICE
from batcher import Batcher, get_input_from_batch
from utils import write_for_rouge, rouge_eval, rouge_log


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
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        # 创建3个目录
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(5)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

#这里要转换输出为数字即output_ids
            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir, batch)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)


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
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            # print(dec_h.size())
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

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    model_path = sys.argv[1]
    beam_processor = BeamSearch(model_path)
    beam_processor.decode()




# """
# decode阶段使用 beam search 算法
# """
# import os
# import sys
# import time
# import torch
# from torch.autograd import Variable

# import data
# import config
# from data import Vocab
# from model import Model
# from config import USE_CUDA, DEVICE
# from batcher import Batcher, get_input_from_batch
# from utils import write_for_rouge, rouge_eval, rouge_log


# class Beam(object):
#     def __init__(self, tokens, log_probs, state, context, coverage):
#         self.tokens = tokens
#         self.log_probs = log_probs
#         self.state = state
#         self.context = context
#         self.coverage = coverage

#     def extend(self, token, log_prob, state, context, coverage):
#         return Beam(tokens = self.tokens + [token],
#                             log_probs = self.log_probs + [log_prob],
#                             state = state,
#                             context = context,
#                             coverage = coverage)

#     @property
#     def latest_token(self):
#         return self.tokens[-1]

#     @property
#     def avg_log_prob(self):
#         return sum(self.log_probs) / len(self.tokens)


# class BeamSearch(object):
#     def __init__(self, model_file_path):
#         model_name = os.path.basename(model_file_path)
#         self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
#         self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
#         self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
#         # 创建3个目录
#         for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
#             if not os.path.exists(p):
#                 os.mkdir(p)

#         self.vocab = Vocab(config.vocab_path, config.vocab_size)
#         self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
#                                batch_size=config.beam_size, single_pass=True)
#         time.sleep(5)

#         self.model = Model(model_file_path, is_eval=True)

#     def sort_beams(self, beams):
#         return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


#     def decode(self):
#         start = time.time()
#         counter = 0
#         batch = self.batcher.next_batch()
#         while batch is not None:
#             # Run beam search to get best Hypothesis
#             best_summary = self.beam_search(batch)

#             # Extract the output ids from the hypothesis and convert back to words
#             output_ids = [int(t) for t in best_summary.tokens[1:]]
#             decoded_words = data.outputids2words(output_ids, self.vocab,
#                                                  (batch.art_oovs[0] if config.pointer_gen else None))

#             # Remove the [STOP] token from decoded_words, if necessary
#             try:
#                 fst_stop_idx = decoded_words.index(data.STOP_DECODING)
#                 decoded_words = decoded_words[:fst_stop_idx]
#             except ValueError:
#                 decoded_words = decoded_words

#             original_abstract_sents = batch.original_abstracts_sents[0]

# #这里要转换输出为数字即output_ids
#             write_for_rouge(original_abstract_sents, output_ids, counter,
#                             self._rouge_ref_dir, self._rouge_dec_dir, batch)
#             counter += 1
#             if counter % 1000 == 0:
#                 print('%d example in %d sec'%(counter, time.time() - start))
#                 start = time.time()

#             batch = self.batcher.next_batch()

#         print("Decoder has finished reading dataset for single_pass.")
#         print("Now starting ROUGE eval...")
#         results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
#         rouge_log(results_dict, self._decode_dir)


#     def beam_search(self, batch):
#         # batch should have only one example
#         enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
#             get_input_from_batch(batch)

#         encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
#         s_t_0 = self.model.reduce_state(encoder_hidden)

#         dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
#         dec_h = dec_h.squeeze()
#         dec_c = dec_c.squeeze()

#         # decoder batch preparation, it has beam_size example initially everything is repeated
#         beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
#                       log_probs=[0.0],
#                       state=(dec_h[0], dec_c[0]),
#                       context = c_t_0[0],
#                       coverage=(coverage_t_0[0] if config.is_coverage else None))
#                  for _ in range(config.beam_size)]
#         results = []
#         steps = 0
#         while steps < config.max_dec_steps and len(results) < config.beam_size:
#             latest_tokens = [h.latest_token for h in beams]
#             latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
#                              for t in latest_tokens]
#             y_t_1 = Variable(torch.LongTensor(latest_tokens))
#             if USE_CUDA:
#                 y_t_1 = y_t_1.to(DEVICE)
#             all_state_h =[]
#             all_state_c = []

#             all_context = []

#             for h in beams:
#                 state_h, state_c = h.state
#                 all_state_h.append(state_h)
#                 all_state_c.append(state_c)

#                 all_context.append(h.context)

#             s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
#             c_t_1 = torch.stack(all_context, 0)

#             coverage_t_1 = None
#             if config.is_coverage:
#                 all_coverage = []
#                 for h in beams:
#                     all_coverage.append(h.coverage)
#                 coverage_t_1 = torch.stack(all_coverage, 0)

#             final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
#                                                         encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
#                                                         extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
#             log_probs = torch.log(final_dist)
#             topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

#             dec_h, dec_c = s_t
#             dec_h = dec_h.squeeze()
#             dec_c = dec_c.squeeze()

#             all_beams = []
#             num_orig_beams = 1 if steps == 0 else len(beams)
#             for i in range(num_orig_beams):
#                 h = beams[i]
#                 state_i = (dec_h[i], dec_c[i])
#                 context_i = c_t[i]
#                 coverage_i = (coverage_t[i] if config.is_coverage else None)

#                 for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
#                     new_beam = h.extend(token=topk_ids[i, j].item(),
#                                    log_prob=topk_log_probs[i, j].item(),
#                                    state=state_i,
#                                    context=context_i,
#                                    coverage=coverage_i)
#                     all_beams.append(new_beam)

#             beams = []
#             for h in self.sort_beams(all_beams):
#                 if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
#                     if steps >= config.min_dec_steps:
#                         results.append(h)
#                 else:
#                     beams.append(h)
#                 if len(beams) == config.beam_size or len(results) == config.beam_size:
#                     break

#             steps += 1

#         if len(results) == 0:
#             results = beams

#         beams_sorted = self.sort_beams(results)

#         return beams_sorted[0]

# if __name__ == '__main__':
#     model_path = sys.argv[1]
#     beam_processor = BeamSearch(model_path)
#     beam_processor.decode()


