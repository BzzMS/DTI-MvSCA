from __future__ import division
from __future__ import print_function

import random
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import GNN
from torch_geometric.loader import ShaDowKHopSampler
from config import Config
from torch_geometric.data import Data
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


cuda_num=0
if __name__ == "__main__":
    config_file = "./config/dti_v3.ini"
    config = Config(config_file)
    fold_ROC = []
    fold_AUPR = []
    criterion = nn.CrossEntropyLoss()
    for fold in range(5):
        config.structgraph_path = "../data_v3/alledg.txt"
        config.train_path = "../data_v3/train00{}.txt".format(fold + 1)
        config.test_path = "../data_v3/test00{}.txt".format(fold + 1)
        # config.structgraph_path = "../data_v5/alledg.txt"
        # config.train_path = "../data_v5/train00{}.txt".format(fold + 1)
        # config.test_path = "../data_v5/test00{}.txt".format(fold + 1)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False  # Close optimization
        torch.backends.cudnn.deterministic = True  # Close optimization

        sadj= load_graph(config)
        features, labels, idx_train, idx_test = load_data(config)

        asadj = get_adj(sadj)
        model = GNN(in_channels=config.fdim,
                    hidden_channels1=config.nhid1,
                    hidden_channels2=config.nhid2,
                    nclass=config.class_num
                    ).cuda(cuda_num)

        features = features
        sadj = sadj
        labels = labels
        idx_train = idx_train
        idx_test = idx_test
        boo_train = index_to_boo(features.shape[0], idx_train)
        boo_test = index_to_boo(features.shape[0], idx_test)
        s_values = sadj._values()
        labels_shape=labels.shape[0]
        tmp_ten=torch.arange(0,labels_shape,1).reshape(labels_shape,1)
        labels=torch.cat((labels.reshape(labels_shape,1),tmp_ten),1)
        s_data = Data(x=features, edge_index=asadj, edge_attr=s_values, y=labels, train_mask=boo_train,
                      test_mask=boo_test).cuda(cuda_num)

        my_depth = 3
        my_num_neighbors = 12
        batch_size=18
        s_train_loader = ShaDowKHopSampler(s_data, depth=my_depth, num_neighbors=my_num_neighbors,
                                           node_idx=s_data.train_mask, shuffle=True, batch_size=batch_size)
        s_test_loader = ShaDowKHopSampler(s_data, depth=my_depth, num_neighbors=my_num_neighbors,
                                          node_idx=s_data.test_mask, shuffle=True)




        optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True,factor=0.8,patience=8,min_lr=0.00003)
        roc = []

        pr = []
        acc = []

        def train(model, epochs):
            model.train()
            num=0
            total_loss=0

            for s_data in tqdm(s_train_loader):

                s_data=s_data.cuda(cuda_num)

                optimizer.zero_grad()
                output = model(s_data.x, s_data.edge_index, s_data.batch, s_data.root_n_id).cuda(cuda_num)

                tmp_lable=s_data.y[:,0]

                loss = criterion(output, tmp_lable)
                loss.backward()
                optimizer.step()
                num=num+1
                total_loss += loss
                real_loss= total_loss / num
            scheduler.step(real_loss)
            return real_loss
        def main_test(model, epoch, loss):
            model.eval()
            outputs=[]
            label=[]
            for s_data in tqdm(s_test_loader):
                s_data = s_data.cuda(cuda_num)
                output = model(s_data.x, s_data.edge_index, s_data.batch, s_data.root_n_id).cuda(cuda_num)
                outputs.append(output.cpu().detach().numpy())
                label.append(int(s_data.y[0][0].cpu()))
            label = np.array(label)
            label = torch.tensor(label)
            outputs = np.array(outputs)
            outputs = torch.tensor(outputs).squeeze()
            c, d = RocAndAupr(outputs, label)
            acc_test = accuracy(outputs, label)
            roc.append(c)
            pr.append(d)
            acc.append(acc_test)

            print("this is the", epoch + 1,
                  "epochs, ROC is {:.4f},and AUPR is {:.4f} test set accuray is {:.4f},loss is {:.4f} ".format(c, d, acc_test,
                                                                                                               loss))


        acc_max = 0
        epoch_max = 0
        roc_max = 0
        pr_max = 0

        for epoch in range(config.epochs):

            loss=train(model, epoch)
            main_test(model, epoch, loss)
            if acc_max < acc[epoch]:
                acc_max = acc[epoch]
            if roc_max < roc[epoch]:
                roc_max = roc[epoch]
            if pr_max < pr[epoch]:
                pr_max = pr[epoch]
            if epoch + 1 == config.epochs:
                fold_ROC.append(roc_max)
                fold_AUPR.append(pr_max)
                print(
                    "this is {} fold ,the max ROC is {:.4f},and max AUPR is {:.4f} test set max  accuray is {:.4f} , ".format(
                        fold, roc_max,
                        pr_max,
                        acc_max))
    print("average AUROC is {:.4} , average AUPR is {:.4}".format(sum(fold_ROC) / len(fold_ROC),
                                                                  sum(fold_AUPR) / len(fold_AUPR)))
