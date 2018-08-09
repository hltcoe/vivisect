import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data
import torch
from torch.autograd import Variable
torch.set_default_tensor_type("torch.DoubleTensor")
import torchtext


class mlp(nn.Module):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(mlp, self).__init__()
        self.dense1 = nn.Linear(in_features=nfeats, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size // 2)
        self.dense3 = nn.Linear(in_features=hidden_size // 2, out_features=hidden_size // 4)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = F.relu(self.dense1(x[0]))
        xx = F.relu(self.dense2(x))
        return self.softmax(F.relu(self.dense3(xx)))

    
class rnn(nn.Module):
    def __init__(self, nfeats, nlabels, hidden_size, rl, cl):
        super(rnn, self).__init__()
        #self.emb = torchtext.vocab.GloVe(dim=300)
        self.lstm = nn.LSTM(input_size=nfeats, hidden_size=hidden_size)
        self.dense = nn.Linear(in_features=hidden_size, out_features=nlabels)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x, l = x
        #print(x.shape, l.shape)
        #sys.exit()
        seq_lengths, perm_idx = l.sort(0, descending=True)
        seq_lengths = seq_lengths.long()
        xx = x[perm_idx]
        #print(xx.shape, seq_lengths)
        xp = pack_padded_sequence(xx, seq_lengths, batch_first=True)
        #print(xp.data.shape, xp.batch_sizes.shape)
        #for l, cs in instancesa:
        #    print(rlabel_lookup[l], "".join([rchar_lookup[c] for c in cs]))

        packed_out, (ht, ct) = self.lstm(xp)
        out, ind = pad_packed_sequence(packed_out, True)
        #print(ind - 1)
        #print(out.shape, ht.squeeze().shape, ct.shape)
        #print(ht.squeeze().shape)
        #print(out.shape)
        #print(ind)
        #inp = torch.Tensor([out[i, v - 1, :] for i, v in enumerate(ind)])
        #out.index_select(1, ind - 1)
        #print(inp.shape)
        #return self.softmax(F.relu(self.dense2(x)))
        return self.softmax(F.relu(self.dense(ht.squeeze())))
        #print(v.shape)
        #return v
        #return (packed_out, (ht, ct))
