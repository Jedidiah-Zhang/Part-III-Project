
from torch import nn 
import torch

class LSTM_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LSTM_Cell, self).__init__()

        self.ix_linear = nn.Linear(in_dim, hidden_dim)
        self.ih_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fx_linear = nn.Linear(in_dim, hidden_dim)
        self.fh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.ox_linear = nn.Linear(in_dim, hidden_dim)
        self.oh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.cx_linear = nn.Linear(in_dim, hidden_dim)
        self.ch_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_1, c_1):
        i = torch.sigmoid(self.ix_linear(x) + self.ih_linear(h_1))
        f = torch.sigmoid(self.fx_linear(x) + self.fh_linear(h_1))
        o = torch.sigmoid(self.ox_linear(x) + self.oh_linear(h_1))
        c_ = torch.tanh(self.cx_linear(x) + self.ch_linear(h_1))

        c = f * c_1 + i * c_
        h = o * torch.tanh(c)

        return h, c
    
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = LSTM_Cell(in_dim, hidden_dim)

    def forward(self, x):
        '''
        x: [seq_lens, batch_size, in_dim]
        '''
        outs = []
        h, c = None, None
        for seq_x in x: # seq_x: [batch_size, in_dim]
            if h is None: h = torch.randn(x.shape[1], self.hidden_dim)
            if c is None: c = torch.randn(x.shape[1], self.hidden_dim)
            h, c = self.lstm_cell(seq_x, h, c)
        outs.append(torch.unsqueeze(h, 0))
        outs = torch.cat(outs)
        return outs, (h, c)
    
if __name__ == "__main__":
    BATCH_SIZE = 24
    SEQ_LENS = 7
    IN_DIM = 12
    OUT_DIM = 6
    
    lstm = LSTM(IN_DIM, OUT_DIM)

    x = torch.randn(SEQ_LENS, BATCH_SIZE, IN_DIM)

    outs, (h, c) = lstm(x)

    print(outs.shape)
    print(h.shape)
    print(c.shape)