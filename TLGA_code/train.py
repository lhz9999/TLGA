import os
import sys
import torch.nn as nn
import torch.nn.functional as F

import jieba
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import config
from data import Vocab
from utils import calc_running_avg_loss
from config import USE_CUDA, DEVICE
from batcher import Batcher
from batcher import get_input_from_batch
from batcher import get_output_from_batch
from adagrad_opt import AdagradCustom


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        train_dir = os.path.join(config.log_root, 'train_{}'.format(stamp))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter_step):
        """保存模型"""
        state = {
            'iter': iter_step,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
        model_save_path = os.path.join(self.model_dir, 'model_{}_{}'.format(iter_step, stamp))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        """模型初始化或加载、初始化迭代次数、损失、优化器"""
        # 初始化模型
        self.model = Model(model_file_path)
        # 模型参数的列表
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        # 定义adam优化器
        # self.optimizer = optim.Adam(params, lr=config.adam_lr)
        # # # 使用AdagradCustom做优化器
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        # 初始化迭代次数和损失
        start_iter, start_loss = 0, 0
        # 如果传入的已存在的模型路径，加载模型继续训练
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location = lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(DEVICE)

        return start_iter, start_loss

    def train_one_batch(self, batch):
       
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)
        self.optimizer.zero_grad()
        
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        s_t_1 = self.model.reduce_state(encoder_hidden)   # (h,c) = ([1, B, hid_dim], [1, B, hid_dim])
        
        in_features = config.hidden_dim * config.batch_size
        out_features = config.hidden_dim * config.batch_size
        alpha = 0.2 # 激活斜率 (LeakyReLU)的激活斜率
        W = nn.Parameter(torch.zeros(size=(in_features, out_features))).cuda() #建立一个w权重，用于对特征数F进行线性变化
        nn.init.xavier_uniform_(W.data, gain=1.414).cuda() #对权重矩阵进行初始化
        aa = nn.Parameter(torch.zeros(size=(2*out_features, 1))).cuda() #计算函数α，输入是上一层两个输出的拼接，输出的是eij，a的size为(2*F',1)
        nn.init.xavier_uniform_(aa.data, gain=1.414).cuda() #对a进行初始化
        leakyrelu = nn.LeakyReLU(alpha).cuda()#激活层



        step_losses = []
        a = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]      # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            # print("y_t_1:", y_t_1, y_t_1.size())
            h_decode, c_decode = s_t_1

            #a.append((h_decode.view(-1)).cpu().detach().numpy().tolist())
            a.append((h_decode.view(-1)).detach().tolist())
            # b.append((c_decode.view(-1)).detach().tolist())
            # print(a)
            # print(b)
#更新h
            gat_input = torch.tensor(a).cuda()
            # print(gat_input.size())
            h_gat = torch.mm(gat_input, W).cuda()#    [N,Feature]
            # print(h_gat.size())
            N = gat_input.size()[0]##获取当前的节点数量
            out_size = torch.zeros(N,config.batch_size,config.hidden_dim).cuda()
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
            # print(h_decoder.size())
# #更新c
#             gat_input_c = torch.tensor(b).cuda()
#             # print(gat_input.size())
#             c_gat = torch.mm(gat_input_c, W).cuda()#    [N,Feature]
#             # print(h_gat.size())
#             N1 = gat_input_c.size()[0]##获取当前的节点数量
#             out_size = torch.zeros(N1,config.batch_size,config.hidden_dim).cuda()
#             c_input = torch.cat([c_gat.repeat(1, N1).view(N1 * N1, -1), c_gat.repeat(N1, 1)], dim=1).view(N1, -1, 2 * out_features).cuda()
#             #每一行是一个隐状态与其他各个隐状态的相关性值
#             e = leakyrelu(torch.matmul(c_input, aa).squeeze(2)).cuda()#[N,N]
#             #做一个softmax，生成贡献度权重
#             decoder_attention_c = F.softmax(e, dim=1).cuda()
#             # print(a_input.size())
#             #根据权重计算最终的特征输出。
#             lstm_out_c = torch.matmul(decoder_attention_c, c_gat).cuda()
#             lstm_out_c = F.elu(lstm_out_c).cuda()

#             lstm_out_c = lstm_out_c.view_as(out_size).cuda()#[len,batch,dim]

#             c_decoder = lstm_out_c[-1,:,:].cuda()#更新decoder隐状态，

#             c_decoder = c_decoder.unsqueeze(0).cuda()
            s_t_1 =  tuple([h_decoder,c_decode])#更新st值

            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                       extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]  # 摘要的下一个单词的编码
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
            step_loss = -torch.log(gold_probs + config.eps)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)    
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        # 训练设置，包括
        iter_step, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        min_loss = 10000
        while iter_step < n_iters:
            # 获取下一个batch数据
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter_step)
            iter_step += 1

            if iter_step % 100 == 0:
                self.summary_writer.flush()
            
            # print_interval = 100
            if iter_step % 100== 0:
                # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('steps %d, seconds for %d steps: %.2f, loss: %f' % (iter_step, 100,
                                                                          time.time() - start, loss))
                start = time.time()

            # 20000次迭代就保存一下模型
            if iter_step %  20000== 0:
                self.save_model(running_avg_loss, iter_step)
                
            if iter_step>10000 and loss < min_loss:
                min_loss = loss
                self.save_model(running_avg_loss, iter_step)

            # if loss <= 0.3 :
            #     self.save_model(running_avg_loss, iter_step)

def init_print():
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("时间:{}".format(stamp))
    print("***参数:***")
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            print(":".join([k, str(v)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    init_print()
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_path)
