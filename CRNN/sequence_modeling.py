import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    Exactly the same as ClovaAI’s:
      - takes (input_size, hidden_size, output_size)
      - runs a bidirectional LSTM of hidden_size → linear(hidden_size*2 → output_size)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # bidirectional LSTM
        self.rnn    = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # map 2*hidden_size → output_size
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: [B, T, input_size]
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)           # [B, T, 2*hidden_size]
        output       = self.linear(recurrent)  # [B, T, output_size]
        return output