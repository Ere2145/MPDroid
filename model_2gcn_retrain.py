import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import dgl
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import LSTM
import numpy as np
import dgl.nn.pytorch as dglnn
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import random
from datetime import datetime

torch.set_printoptions(profile="full")

class MultiGraphDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.labels = []
        self.lengths = []
        self.median_length=0
        

    def generate(self, apig_path, fcg_path):
        print("start generate dataset")
        start=datetime.now()
        total=0
        for label, name in enumerate(['bni', 'mal']):
            cnt=0
            for file in os.listdir(os.path.join(apig_path, name)):
                filename = file.replace('.csv', '').replace('.txt', '').replace('.apig','')
                fcg_file = os.path.join(fcg_path, name, filename+'.fcg')
                if not os.path.exists(fcg_file):
                    continue
                cnt+=1
                print(cnt,filename)
                
                apig ,_= dgl.load_graphs(os.path.join(apig_path, name, file),[0])
                apig=apig[0]
                if apig.num_nodes()==0:
                    with open('dataset_load_logfile.txt','a') as f:
                        print(f"apig {file} got 0 nodes")
                        f.write(f"apig {file} got 0 nodes\n")
                        continue
                old_feat = apig.ndata['feat']
                new_feat = torch.zeros((len(old_feat), 115))
                for i, val in enumerate(old_feat):
                    new_feat[i][i] = val
                apig.ndata['feat'] = new_feat
                apig=dgl.add_self_loop(apig)

                
                fcg ,_= dgl.load_graphs(os.path.join(fcg_path, name, filename+'.fcg'),[0])
                fcg=fcg_preprocess(fcg[0])
                if fcg.num_nodes()==0:
                    with open('dataset_load_logfile.txt','a') as f:
                        print(f"fcg {file} got 0 nodes")
                        f.write(f"fcg {file} got 0 nodes\n")
                        continue
                if fcg is not None:
                    self.samples.append((apig, fcg))
                else:
                    print(f"Failed to load graph for file {file}")
                    exit(0)
                self.labels.append(label)
            total+=cnt+1
        print("end generate dataset")
        end=datetime.now()
        print("start ",start,"end ",end,"total ",total)

    def preApig(self):
        for apig,_ in self.samples:
                old_feat = apig.ndata['feat']
                new_feat = torch.zeros((len(old_feat), 115))
                for i, val in enumerate(old_feat):
                    new_feat[i][i] = val
                apig.ndata['feat'] = new_feat
                apig=dgl.add_self_loop(apig)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump((self.samples, self.labels), f)

    def load(self,path):
        with open(path, 'rb') as f:
            self.samples, self.labels = pickle.load(f)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, apig_embedding, fcg_embedding):
        """z-score"""
        # 对apig_embedding进行标准化
        apig_embedding = (apig_embedding - apig_embedding.mean()) / apig_embedding.std()
        apig_embedding = (apig_embedding - apig_embedding.min()) / (apig_embedding.max() - apig_embedding.min())
        # 对fcg_embedding进行标准化
        fcg_embedding = (fcg_embedding - fcg_embedding.mean()) / fcg_embedding.std()
        fcg_embedding = (fcg_embedding - fcg_embedding.min()) / (fcg_embedding.max() - fcg_embedding.min())
        attn_weights = self.softmax(self.attn(apig_embedding))
        attn_api = torch.mul(apig_embedding, attn_weights)
        attn_weights = self.softmax(self.attn(fcg_embedding))
        attn_fcg = torch.mul(fcg_embedding, attn_weights)
        attn_applied=attn_api+attn_fcg
        #attn_applied=torch.cat((attn_api,attn_fcg),0)
        return attn_applied

#region newmodel
class MultiGraphClassifier(nn.Module):
    def __init__(self, apig_dim, fcg_dim, hidden_dim, num_classes):
        super(MultiGraphClassifier, self).__init__()
        self.conv_apig1 = dglnn.GraphConv(apig_dim, hidden_dim)
        self.conv_apig2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv_fcg1 = dglnn.GraphConv(fcg_dim, hidden_dim)
        self.conv_fcg2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.dropout=nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, apig=None, apig_feat=None, fcg=None, fcg_feat=None):
        h_a1 = F.relu(self.conv_apig1(apig, apig_feat))
        h_a2 = F.relu(self.conv_apig2(apig, h_a1))
        h_f1 = F.relu(self.conv_fcg1(fcg, fcg_feat))
        h_f2 = F.relu(self.conv_fcg2(fcg, h_f1))
        with apig.local_scope():
            apig.ndata['h'] = h_a2
            apig_embedding = dgl.mean_nodes(apig, 'h').reshape(1,-1).squeeze()
        with fcg.local_scope():
            fcg.ndata['h'] = h_f2
            fcg_embedding = dgl.max_nodes(fcg, 'h').reshape(1,-1).squeeze()
        embedding = self.attention(apig_embedding, fcg_embedding)
        embedding=self.dropout(embedding)
        output = self.classifier(embedding)
        for val in output:
            if np.isnan(val.cpu().detach().numpy()):
                print(f"nan occured")
                exit(0)
        return output

#endregion

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = zip(*samples)
    apigs=[]
    fcgs=[]
    for data in graphs:
        apigs.append(data[0])
        fcgs.append(data[1])
    labels = torch.tensor(labels).to('cuda')  # 将标签转移到 GPU
    return apigs, fcgs, labels

if __name__=="__main__":
    dataset=MultiGraphDataset()
    #dataset.generate('./api_src/api_graph','./fcg_src/fcg_total')
    #dataset.save('./dataset/multifeatDataset_2graph_1.pkl')
    dataset.load('./dataset/multifeatDataset_2graph_2.pkl')
    #dataset.preApig()
    #dataset.save('./dataset/multifeatDataset_2graph_2.pkl')
    print("-------------------finish load dataset-------------------")
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    model = MultiGraphClassifier(apig_dim=115, fcg_dim=226, hidden_dim=64, num_classes=2)
    model = model.to('cuda')

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    losses=[]
    # 训练模型
    print("-------------------train begin at ",datetime.now(),"-------------------")
    model.train()
    for epoch in range(100):
        total = 0
        correct = 0
        for i, (apigs_s, fcgs_s, labels) in enumerate(train_loader):
            #graphs=[fcg_mask(graph,random.randint(0,2)).to('cuda') for graph in graphs]
            apigs=[dgl.add_self_loop(graph).to('cuda') for graph in apigs_s]
            apig_feats=[]
            for apig in apigs:
                apig_feats.append(apig.ndata['feat'].to('cuda'))
            
            #fcgs=[dgl.add_self_loop(graph).to('cuda') for graph in fcgs_s]
            #fcg_feats=[]
            #for fcg in fcgs:
            #    fcg_feats.append(fcg.ndata['api'].to('cuda'))
            
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = [model(apig, apig_feat) for apig, apig_feat in zip(apigs, apig_feats)]
            outputs=torch.stack(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 得到分类结果
            _, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()

        accuracy = round(correct / total, 3)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}')
    print("-------------------train end at ",datetime.now(),"-------------------")