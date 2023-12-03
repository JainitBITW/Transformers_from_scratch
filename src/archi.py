import torch 
from torch import nn
import torch.nn.functional as F
import math
import copy
import re 
import pytorch_lightning as pl
from torch.utils.data import DataLoader , Dataset
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 

def smoothedBleu(reference, candidate):
    smoothing_function = SmoothingFunction().method7
    return sentence_bleu([reference], candidate, smoothing_function=smoothing_function)


class FeedForward(nn.Module):
    def __init__(self , d_model , d_ff , dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model , d_ff)
        self.linear2 = nn.Linear(d_ff , d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self , x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self , d_model , num_heads , dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads
        self.linear_q = nn.Linear(d_model , d_model)
        self.linear_k = nn.Linear(d_model , d_model)
        self.linear_v = nn.Linear(d_model , d_model)
        self.linear_o = nn.Linear(d_model , d_model)
    
    def forward(self , query , key , value , mask = None):
        batch_size = query.size(0)
        #query = [batch_size , seq_len , d_model]
        #key = [batch_size , seq_len , d_model]
        #value = [batch_size , seq_len , d_model]
        #mask = [batch_size , seq_len , seq_len]
        
        #linear transformation
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        #query = [batch_size , seq_len , d_model]
        #key = [batch_size , seq_len , d_model]
        #value = [batch_size , seq_len , d_model]
        
        #split the query , key and value into num_heads
        query = query.view(batch_size , -1 , self.num_heads , self.d_k)
        key = key.view(batch_size , -1 , self.num_heads , self.d_k)
        value = value.view(batch_size , -1 , self.num_heads , self.d_k)
        #query = [batch_size , seq_len , num_heads , d_k]
        #key = [batch_size , seq_len , num_heads , d_k]
        #value = [batch_size , seq_len , num_heads , d_k]
        
        #transpose the query , key and value
        query = query.transpose(1 , 2)
        key = key.transpose(1 , 2)
        value = value.transpose(1 , 2)
        #query = [batch_size , num_heads , seq_len , d_k]
        #key = [batch_size , num_heads , seq_len , d_k]
        #value = [batch_size , num_heads , seq_len , d_k]
        
        #calculate the scores
        scores = torch.matmul(query , key.transpose(-2 , -1))
        #scores = [batch_size , num_heads , seq_len , seq_len]
        
        #scale the scores
        scores = scores / math.sqrt(self.d_k)

        #apply softmax
        if mask is not None:
            
            mask = mask.unsqueeze(1)
            mask = mask.expand_as(scores)
            mask.to(scores.device)
            scores = scores.masked_fill(mask == 0 , -1e9)
            
        scores = F.softmax(scores , dim = -1)
        scores = self.dropout(scores)
        #scores = [batch_size , num_heads , seq_len , seq_len]
        
        #apply attention
        context = torch.matmul(scores , value)
        #context = [batch_size , num_heads , seq_len , d_k]
        
        #concat the heads
        context = context.transpose(1 , 2)
        #context = [batch_size , seq_len , num_heads , d_k]
        context = context.contiguous().view(batch_size , -1 , self.d_model)
        #context = [batch_size , seq_len , d_model]
        
        #linear transformation
        context = self.linear_o(context)
        #context = [batch_size , seq_len , d_model]
        return context

        
class LayerNorm(nn.Module):
    def __init__(self , d_model , eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(self.d_model))
        self.b_2 = nn.Parameter(torch.zeros(self.d_model))
    
    def forward(self , x):
        mean = x.mean(-1 , keepdim = True)
        std = x.std(-1 , keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model , num_heads , d_ff , num_layers , dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.multi_head_attention = MultiHeadAttention(d_model , num_heads , dropout)
        self.feed_forward = FeedForward(d_model , d_ff , dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        
    def forward(self , x , mask = None):
        batch, seq_len, d_model = x.size()
        if mask is not None: 
            mask = mask.unsqueeze(1)
            mask = mask.expand(batch, seq_len, seq_len)
        
        x_m = self.multi_head_attention(x , x , x , mask)
        x = self.dropout(x)
        x = self.layer_norm1(x + x_m)
        x_f = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + x_f)
        return x
    
        
        #x = [batch_size , seq_len , d_model]
        #mask = [batch_size , seq_len , seq_len]
        

class DecoderLayer(nn.Module):
    def __init__(self, d_model , num_heads , d_ff , num_layers , dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.multi_head_attention1 = MultiHeadAttention(d_model , num_heads , dropout)
        self.multi_head_attention2 = MultiHeadAttention(d_model , num_heads , dropout)
        self.feed_forward = FeedForward(d_model , d_ff , dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
    
    def forward(self , x , encoder_output,src_mask=None ,tgt_mask = None):
        #x = [batch_size , seq_len , d_model]
        #encoder_output = [batch_size , seq_len , d_model]
        #src_mask = [batch_size , seq_len ]
        batch_size = x.size(0)
        seq_len = x.size(1)
        look_ahead_mask = torch.tril(torch.ones(x.size(1) , x.size(1))).unsqueeze(0)
        look_ahead_mask = look_ahead_mask.to(x.device)
        x_m = self.multi_head_attention1(x , x , x , look_ahead_mask)
        x = self.dropout(x)
        x = self.layer_norm1(x + x_m)
        cross_mask = src_mask.unsqueeze(1).expand(batch_size , seq_len , src_mask.size(1)) 
        cross_mask = cross_mask.to(x.device)
        x_m = self.multi_head_attention2(x , encoder_output , encoder_output , cross_mask)
        x = self.dropout(x)
        x = self.layer_norm2(x + x_m)
        x_f = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm3(x + x_f)
        return x
    
        
        
class Encoder(nn.Module):
    def __init__(self , d_model , num_heads , d_ff , num_layers , dropout = 0.1 , en_embedding = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model , num_heads , d_ff , num_layers , dropout) for _ in range(num_layers)])
        self.en_embedding = en_embedding
    
    def forward(self , x , mask = None):
        x = self.en_embedding(x)
        for layer in self.layers:
            x = layer(x , mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self , d_model , num_heads , d_ff , num_layers , dropout = 0.1 , fr_embedding = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.fr_embedding = fr_embedding
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model , num_heads , d_ff , num_layers , dropout) for _ in range(num_layers)])
    
    def forward(self , x , encoder_output , src_mask = None , tgt_mask = None):
        x = self.fr_embedding(x)
        for layer in self.layers:
            x = layer(x , encoder_output , src_mask , tgt_mask)
        return x
    
    

class Transformer(nn.Module):
    def __init__(self , d_model , num_heads , d_ff , num_layers , dropout = 0.1 , en_vocab = en_vocab , fr_vocab = fr_vocab):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.en_embedding = nn.Embedding(len(en_vocab) , d_model)
        self.fr_embedding = nn.Embedding(len(fr_vocab) , d_model)
        self.encoder = Encoder(d_model , num_heads , d_ff , num_layers , dropout, self.en_embedding)
        self.decoder = Decoder(d_model , num_heads , d_ff , num_layers , dropout, self.fr_embedding)
        self.linear = nn.Linear(d_model , len(fr_vocab))
    
    def forward(self , src , tgt , src_mask = None , tgt_mask = None):
        
        encoder_output = self.encoder(src , src_mask)
        decoder_output = self.decoder(tgt , encoder_output , src_mask , tgt_mask)
        
        output = self.linear(decoder_output)
        return output
    
    def translate(self , src , src_mask = None , max_len = 100):
        
        encoder_output = self.encoder(src , src_mask)
        tgt = torch.zeros(src.size(0) , 1).long().to(src.device)
        for i in range(max_len):
            tgt_mask = torch.tril(torch.ones(tgt.size(1) , tgt.size(1))).unsqueeze(0).to(src.device)
            decoder_output = self.decoder(tgt , encoder_output , src_mask , tgt_mask)
            output = self.linear(decoder_output)
            output = output[: , -1 , :]
            output = torch.argmax(output , dim = -1)
            tgt = torch.cat([tgt , output.unsqueeze(1)] , dim = 1)
        return tgt


class LitTransformer(pl.LightningModule):
    def __init__(self, d_model , num_heads , d_ff , num_layers , dropout , en_vocab , fr_vocab):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.transformer = Transformer(d_model , num_heads , d_ff , num_layers , dropout , en_vocab , fr_vocab)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.fr_vocab.index("<pad>"))
        
    def forward(self , src , tgt , src_mask = None , tgt_mask = None):
        return self.transformer(src , tgt , src_mask , tgt_mask)
    
    def training_step(self , batch , batch_idx):
        src , tgt = batch
        src_input_ids , src_attention_mask = src
        tgt_input_ids , tgt_attention_mask = tgt
        output = self(src_input_ids , tgt_input_ids[:,:-1] , src_attention_mask , tgt_attention_mask[:,:-1])
        translated = output.argmax(dim = -1)
        translated = translated.detach().cpu().numpy()
        translated = [self.fr_vocab.itos[idx] for idx in translated[0]]
        self.logger.experiment.add_text("translated" , " ".join(translated) , self.current_epoch)
        bleu = smoothedBleu([[translated]] , [[self.fr_vocab.itos[idx] for idx in tgt_input_ids[0].tolist() if idx != self.fr_vocab.index("<pad>")]])
        self.logger.experiment.add_scalar("bleu" , bleu , self.current_epoch)
        output = output.contiguous().view(-1 , output.size(-1))
        tgt_input_ids = tgt_input_ids[:,1:].contiguous().view(-1)
        loss = self.criterion(output , tgt_input_ids)
        self.log("train_loss" , loss)
        return loss
    
    def validation_step(self , batch , batch_idx):
        src , tgt = batch
        src_input_ids , src_attention_mask = src
        tgt_input_ids , tgt_attention_mask = tgt
        output = self(src_input_ids , tgt_input_ids[:,:-1] , src_attention_mask , tgt_attention_mask[:,:-1])
        translated = output.argmax(dim = -1)
        translated = translated.detach().cpu().numpy()
        translated = [self.fr_vocab.itos[idx] for idx in translated[0]]
        bleu = smoothedBleu([[translated]] , [[self.fr_vocab.itos[idx] for idx in tgt_input_ids[0].tolist() if idx != self.fr_vocab.index("<pad>")]])
        self.logger.experiment.add_scalar("bleu" , bleu , self.current_epoch)
        self.logger.experiment.add_text("translated" , " ".join(translated) , self.current_epoch)
        output = output.contiguous().view(-1 , output.size(-1))
        tgt_input_ids = tgt_input_ids[:,1:].contiguous().view(-1)
        
        loss = self.criterion(output , tgt_input_ids)
        self.log("val_loss" , loss)
        return loss
    
    def test_step(self , batch , batch_idx):
        src , tgt = batch
        src_input_ids , src_attention_mask = src
        tgt_input_ids , tgt_attention_mask = tgt
        output = self(src_input_ids , tgt_input_ids[:,:-1] , src_attention_mask , tgt_attention_mask[:,:-1])
        translated = output.argmax(dim = -1)
        translated = translated.detach().cpu().numpy()
        translated = [self.fr_vocab.itos[idx] for idx in translated[0]]
        bleu = smoothedBleu([[translated]] , [[self.fr_vocab.itos[idx] for idx in tgt_input_ids[0].tolist() if idx != self.fr_vocab.index("<pad>")]])
        self.logger.experiment.add_scalar("bleu" , bleu , self.current_epoch)
        self.logger.experiment.add_text("translated" , " ".join(translated) , self.current_epoch)
        output = output.contiguous().view(-1 , output.size(-1))
        tgt_input_ids = tgt_input_ids[:,1:].contiguous().view(-1)
        loss = self.criterion(output , tgt_input_ids)
        self.log("test_loss" , loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters() , lr = 0.0001)
    
    def translate(self , src , src_mask = None , max_len = 100):
        return self.transformer.translate(src , src_mask , max_len)
    