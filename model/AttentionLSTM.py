import torch
import torch.nn as nn

# 定义带有注意力机制的LSTM模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = nn.LSTM(configs.enc_in, configs.d_model, configs.e_layers, batch_first=True)
        self.decoder = nn.LSTM(configs.dec_in, configs.d_model, configs.d_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim = configs.d_model, num_heads=configs.n_heads, batch_first = True)
        self.fc = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, input_seq, output_seq):
        # 编码器处理输入序列
        encoder_output, _ = self.encoder(input_seq)

        # 解码器处理输出序列
        decoder_output, _ = self.decoder(output_seq)

        # 计算注意力权重
        attn_output, _ = self.attention(decoder_output, encoder_output, encoder_output)

        # 使用注意力权重生成输出序列
        output = self.fc(attn_output)

        return output

input_data_inference = torch.randn(64, 1000, 100)
import argparse

parser = argparse.ArgumentParser(description='Command-line parser example')
parser.add_argument('--e_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--d_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--d_model', type=int, default=128, help='Hidden size')
parser.add_argument('--enc_in', type=int, default=100, help='Encoder input size')
parser.add_argument('--dec_in', type=int, default=100, help='Decoder input size')
parser.add_argument('--c_out', type=int, default=10, help='Output size')
parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')


args = parser.parse_args()
model = Model(args)
output_sequence = model(input_data_inference, output_seq=torch.randn(64, 1000, 100))

print("Inference Result:")
print(output_sequence.shape)