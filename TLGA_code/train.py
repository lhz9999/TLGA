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
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())

        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

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
        alpha = 0.2 
        W = nn.Parameter(torch.zeros(size=(in_features, out_features))).cuda() 
        nn.init.xavier_uniform_(W.data, gain=1.414).cuda() 
        aa = nn.Parameter(torch.zeros(size=(2*out_features, 1))).cuda() 
        nn.init.xavier_uniform_(aa.data, gain=1.414).cuda() 
        leakyrelu = nn.LeakyReLU(alpha).cuda()



        step_losses = []
        a = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]      

            h_decode, c_decode = s_t_1

            a.append((h_decode.view(-1)).detach().tolist())

            gat_input = torch.tensor(a).cuda()
            h_gat = torch.mm(gat_input, W).cuda()#    [N,Feature]
            N = gat_input.size()[0]
            out_size = torch.zeros(N,config.batch_size,config.hidden_dim).cuda()
            a_input = torch.cat([h_gat.repeat(1, N).view(N * N, -1), h_gat.repeat(N, 1)], dim=1).view(N, -1, 2 * out_features).cuda()
            e = leakyrelu(torch.matmul(a_input, aa).squeeze(2)).cuda()#[N,N]
            decoder_attention = F.softmax(e, dim=1).cuda()

            lstm_out_attention = torch.matmul(decoder_attention, h_gat).cuda()
            lstm_out_attention = F.elu(lstm_out_attention).cuda()

            lstm_out_attention = lstm_out_attention.view_as(out_size).cuda()#[len,batch,dim]

            h_decoder = lstm_out_attention[-1,:,:].cuda()

            h_decoder = h_decoder.unsqueeze(0).cuda()
           
            s_t_1 =  tuple([h_decoder,c_decode])#更新st值

            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                       extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di] 
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   
            step_loss = -torch.log(gold_probs + config.eps) 
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
        iter_step, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        min_loss = 10000
        while iter_step < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter_step)
            iter_step += 1

            if iter_step % 100 == 0:
                self.summary_writer.flush()
            
            if iter_step % 100== 0:

                print('steps %d, seconds for %d steps: %.2f, loss: %f' % (iter_step, 100,
                                                                          time.time() - start, loss))
                start = time.time()

            if iter_step %  20000== 0:
                self.save_model(running_avg_loss, iter_step)
                
            if iter_step>10000 and loss < min_loss:
                min_loss = loss
                self.save_model(running_avg_loss, iter_step)

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
