import scipy.sparse as sp
import torch
import numpy as np
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()

    correct = correct.sum()
    return correct / len(labels)


def RocAndAupr(output, labels):
    predict = []

    for i in range(len(output)):
        a = output[i].detach()
        predict.append(float(a[1]))
    predict=np.array(predict)
    labels=labels.cpu()
    c = roc_auc_score(labels, predict)
    precision, recall, thresholds = precision_recall_curve(labels, predict)
    d = auc3(recall, precision)
    return c, d


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(config):
    f = np.loadtxt(config.feature_path, dtype=float)
    l = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test


def load_graph(config):
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return nsadj


def get_adj(adj):
    rows = []
    cols = []
    for i in range(len(adj)):
        col = adj[i].coalesce().indices().numpy()[0]
        for j in range(len(col)):
            rows.append(i)
            cols.append(col[j])
    edge_index = torch.Tensor([rows, cols]).long()
    return edge_index

def index_to_boo(dim,index):
    boo=[False]*dim
    for i in index:
        boo[i]=True
    boo=torch.tensor(boo)
    return boo