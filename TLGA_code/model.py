import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
from config import USE_CUDA, DEVICE
from Transformer import *
from torchinfo import summary
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
   
    def __init__(self,
               vocab_size=config.vocab_size,
               max_seq_len=config.max_enc_steps,
               numlayers=6,
               model_dim=config.emb_dim,
               num_heads=8,
               ffn_dim=1024,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(numlayers)])

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.feature = nn.Linear(config.hidden_dim*2, config.hidden_dim*2, bias=False)
        self.lstm = nn.LSTM(model_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
       
    def forward(self, inputs, inputs_len):
        output = self.embedding(inputs)
        output += self.pos_embedding(inputs_len)
    
        self_attention_mask = padding_mask(inputs, inputs)
    
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        

        packed = pack_padded_sequence(output, inputs_len, batch_first=True)
        lstmoutput, hidden = self.lstm(packed)  # hidden is tuple([2, batch, hid_dim], [2, batch, hid_dim])


        encoder_outputs, _ = pad_packed_sequence(lstmoutput, batch_first=True)
      
        encoder_outputs = encoder_outputs.contiguous()
        encoder_feature = encoder_outputs.view(-1, config.hidden_dim*2)  
        encoder_feature = self.feature(encoder_feature)       

        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden    # h, c dim = [2, batch, hidden_dim]
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)  
        hidden_reduced_h = F.relu(self.reduce_h(h_in))                        
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)             
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() 
        dec_fea_expanded = dec_fea_expanded.view(-1, n)   

        att_features = encoder_feature + dec_fea_expanded 
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)         
            coverage_feature = self.W_c(coverage_input)   
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)   
        scores = self.v(e)             
        scores = scores.view(-1, t_k)  

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask 
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)          
        c_t = torch.bmm(attn_dist, encoder_outputs)  
        c_t = c_t.view(-1, config.hidden_dim * 2)    

        attn_dist = attn_dist.view(-1, t_k)         

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
       
        self.embedding = nn.Embedding(config.vocab_size+1, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)
        

    def forward(self, y_t_1, s_t_1,encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1

            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  
           
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
       
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t

        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) 
        
        output = self.out2(output) 
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if USE_CUDA:
            encoder = encoder.to(DEVICE)
            decoder = decoder.to(DEVICE)
            reduce_state = reduce_state.to(DEVICE)
       
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
        summary(self.encoder)
        summary(self.decoder)
        summary(self.reduce_state)

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
