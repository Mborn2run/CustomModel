import torch
import torch.nn as nn
import random

class Model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = target.shape
        target_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, seq_len, target_vocab_size).to(self.device)
        encoder_output, (hidden, cell) = self.encoder(source)
        input = target[:, 0, :].unsqueeze(1)

        for t in range(1, seq_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            best_guess = output.argmax(2)
            input = target[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else best_guess

        return outputs
    

class Seq2SeqEncoder(nn.Module):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, seq_len, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.lstm = nn.LSTM(embed_size, num_hiddens, num_layers, batch_first=True,
                          dropout=dropout)

    def forward(self, X):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        output, (hidden, cell) = self.lstm(X)
        return output, (hidden, cell)
    

class Seq2SeqDecoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dense = nn.Linear(hidden_size, output_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, hidden, cell):
        output, (hidden, cell) = self.lstm(X, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell
    
seq2SeqEncoder = Seq2SeqEncoder(vocab_size=10, embed_size=80, num_hiddens=16, num_layers=2)
seq2SeqDecoder = Seq2SeqDecoder(vocab_size=10, embed_size=80, num_hiddens=16, num_layers=2)
x = torch.randn(4, 7, 10)
output, state = seq2SeqEncoder(x)
print(output.shape)
print(state.shape)
output_de, state = seq2SeqDecoder(x, state)
print(output_de.shape)