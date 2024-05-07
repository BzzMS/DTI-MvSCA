import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv
from torch.nn import AdaptiveAvgPool1d,LeakyReLU
import copy

'''
in_channels 500, hidden_channels 256 64, out_channels 2
'''


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, nclass):
        super().__init__()
        head = 1
        self.cnn0 = nn.Conv1d(in_channels=in_channels, out_channels=350, kernel_size=1)
        self.gatv1 = GATv2Conv(350, hidden_channels1, heads=head, dropout=0.2)
        self.gatv2 = GATv2Conv(head * hidden_channels1, int(hidden_channels1 / 2), heads=head, dropout=0.2)
        self.gatv3 = GATv2Conv(head * int(hidden_channels1 / 2), hidden_channels2, heads=1, dropout=0.2)
        self.self_attention = SelfAttention(in_channels, 350)

        kernel_size = 3
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size)
        self.cnn2 = nn.Conv1d((in_channels - kernel_size + 1), hidden_channels1, kernel_size=1)
        self.cnn3 = nn.Conv1d(hidden_channels1, int(hidden_channels1 / 2), kernel_size=1)

        self.pool = AdaptiveAvgPool1d(hidden_channels2)
        self.lin = torch.nn.Linear(2 * hidden_channels2, hidden_channels2)  # 128 64
        self.attention = Attention(hidden_channels2)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_channels2, 16),
            nn.Tanh(),
            nn.Linear(16, nclass),
            nn.Softmax(dim=1)
        )

    def forward(self, s_x, s_edge_index, s_batch, s_root_n_id):
        s_x_info = copy.deepcopy(s_x)

        s_x=(s_x).unsqueeze(1)
        s_x=(self.self_attention(s_x)).squeeze(1)
        s_x = self.gatv1(s_x, s_edge_index).relu()
        s_x = self.gatv2(s_x, s_edge_index).relu()
        s_x = self.gatv3(s_x, s_edge_index).relu()
        s_x = torch.cat([s_x[s_root_n_id], global_mean_pool(s_x, s_batch)], dim=-1)


        s_info = s_x_info[s_root_n_id].unsqueeze(1)
        s_info = self.cnn1(s_info)
        s_info = F.dropout(s_info, p=0.2)
        s_info = s_info.squeeze(1)
        s_info = F.relu(self.cnn2(s_info.T))
        s_info = F.dropout(s_info, p=0.2)
        s_info = F.relu(self.cnn3(s_info))
        s_info = F.dropout(s_info, p=0.2)
        s_info = F.relu(self.pool(s_info.T))



        s_x = F.relu(self.lin(s_x))
        emb = torch.stack([s_x, s_info], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (
                self.hidden_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_probs, values)

        return context


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
