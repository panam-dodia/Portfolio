import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch size, src len, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # hidden = [batch size, src len, dec hid dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]
        
        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim + emb_dim + hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch size, 1]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim]
        
        input = input.unsqueeze(1)  # Add sequence length dimension
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        
        a = self.attention(hidden[-1], encoder_outputs)
        # a = [batch size, src len]
        
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]
        
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [batch size, 1, emb dim + enc hid dim]
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output = [batch size, seq len, dec hid dim]
        # hidden = [n layers, batch size, dec hid dim]
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # tgt = [batch size, tgt len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len-1, tgt_vocab_size).to(self.device)
        
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> tokens
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t-1] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = tgt[:, t] if teacher_force else top1
            
        return outputs

def create_model(src_vocab_size, tgt_vocab_size, device):
    INPUT_DIM = src_vocab_size
    OUTPUT_DIM = tgt_vocab_size
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attn = Attention(HID_DIM, HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
                
    model.apply(init_weights)
    
    return model