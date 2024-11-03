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

def fcg_preprocess(g):
    #print("before preprocess: ",g.num_nodes(),torch.sum(g.ndata['api']))
    #g=g.to('cuda')
    api=torch.sum(g.ndata['api'],dim=1)
    node_to_rmv=torch.nonzero(api==0).squeeze()
    node_to_rmv_l=node_to_rmv.tolist()
    aggr_node_id=g.num_nodes()
    aggr_node_edge_src_l=[]
    aggr_node_edge_dst_l=[]
    for idx in node_to_rmv_l:
        frontier=g.sample_neighbors(idx,-1)
        neighbor_t=frontier.edges()
        for i in range(neighbor_t[0].shape[0]):
            src=neighbor_t[0][i]
            dst=neighbor_t[1][i]
            if(src==idx):
                aggr_node_edge_src_l.append(aggr_node_id)
            else:
                aggr_node_edge_src_l.append(src)
            if(dst==idx):
                aggr_node_edge_dst_l.append(aggr_node_id)
            else:
                aggr_node_edge_src_l.append(dst)
    aggr_node_edge_src_t=torch.tensor(aggr_node_edge_src_l,dtype=int)
    aggr_node_edge_dst_t=torch.tensor(aggr_node_edge_dst_l,dtype=int)
    g.add_edges(aggr_node_edge_src_t,aggr_node_edge_dst_t,etype='_E')
    g.remove_nodes(node_to_rmv)
    """聚合节点"""
    rolls=0
    alpha=0.5
    k=0.3
    max_size=1000
    while(g.num_nodes()>max_size):
        #计算节点优先级，考虑敏感api调用数量以及节点出度;对重要节点和不重要的topK均聚合,聚合权重暂时设为邻居数量的倒数
        #print("rolls {},nodes {}".format(rolls,g.num_nodes(),g.num_edges()))
        pri_score=torch.zeros(g.num_nodes())
        out_degrees=g.out_degrees()
        api=torch.sum(g.ndata['api'],dim=1)
        pri_score=alpha*api+(1-alpha)*out_degrees
        k=int(k*g.num_nodes())
        _,indices_t=pri_score.topk(k=k,dim=0,largest=True)
        indices_l=indices_t.tolist()
        node_to_rmv=[]
        for idx in indices_l:
            frontier=g.sample_neighbors(idx,-1)
            neighbor_t=frontier.edges()
            neighbor_l=neighbor_t[0].tolist()+neighbor_t[1].tolist()
            n_cnt=len(neighbor_l)
            for n_idx in neighbor_l:
                if n_idx in indices_l:
                    continue
                g.ndata['api'][idx]+=g.ndata['api'][n_idx]/n_cnt
                """添加到二阶邻居的边，删除一阶邻居"""
                frontier_2nd=g.sample_neighbors(n_idx,-1)
                n_2nd_t=frontier_2nd.edges()
                n_2nd_l=n_2nd_t[0].tolist()+n_2nd_t[1].tolist()
                n_2nd_s=set()
                for n_2nd in n_2nd_l:
                    n_2nd_s.add(n_2nd)
                for n_2nd_idx in n_2nd_s:
                    if n_2nd_idx!=idx and n_2nd_idx!=n_idx:
                        g.add_edges(idx,n_2nd_idx)
                node_to_rmv.append(n_idx)
        _,indices_t=pri_score.topk(k=k,dim=0,largest=False)
        indices_l=indices_t.tolist()
        for idx in indices_l:
            frontier=g.sample_neighbors(idx,-1)
            neighbor_t=frontier.edges()
            neighbor_l=neighbor_t[0].tolist()+neighbor_t[1].tolist()
            n_cnt=len(neighbor_l)
            for n_idx in neighbor_l:
                if n_idx in indices_l:
                    continue
                g.ndata['api'][idx]+=g.ndata['api'][n_idx]/n_cnt
                """添加到二阶邻居的边，删除一阶邻居"""
                frontier_2nd=g.sample_neighbors(n_idx,-1)
                n_2nd_t=frontier_2nd.edges()
                n_2nd_l=n_2nd_t[0].tolist()+n_2nd_t[1].tolist()
                n_2nd_s=set()
                for n_2nd in n_2nd_l:
                    n_2nd_s.add(n_2nd)
                #print(idx,n_idx,n_2nd_s)
                for n_2nd_idx in n_2nd_s:
                    if n_2nd_idx!=idx and n_2nd_idx!=n_idx:
                        g.add_edges(idx,n_2nd_idx)
                node_to_rmv.append(n_idx)
        #print(node_to_rmv)
        if(len(node_to_rmv)==0):
            break
        node_to_rmv=torch.tensor(node_to_rmv,dtype=int)
        g.remove_nodes(node_to_rmv)
        rolls+=1
    g=dgl.add_self_loop(g,etype='_E')
    #print("after preprocess: ",g.num_nodes(), torch.sum(g.ndata['api']))
    return g
    
def fcg_mask(graph,type):
    #type:
    # 0:featMask
    # 1:nodeDrop
    # 2:edgeRewire
    #print(f'fcg_mask type:{type}(graph:{graph.num_nodes()} nodes,{graph.num_edges()} edges)')
    #特征遮挡
    if type==0:
        s=set()
        feat_name='api'
        #随机将5%个节点的特征置零
        num=graph.num_nodes()//20
        while(len(s)<num):
            rand_idx=torch.randint(0,graph.num_nodes(),[1,1]).reshape(-1)
            new_feat=torch.zeros(graph.ndata[feat_name][rand_idx].shape)
            #print('new_feat: ',new_feat)
            #exit(0)
            graph.ndata[feat_name][rand_idx]=new_feat
            s.add(rand_idx)
        return graph
    #节点丢弃
    elif type==1:
        #采用完全移除节点特征和结构的方式，为了保持输入维度的一致
        #随机移除5%个节点
        #removed=torch.tensor(np.random.randint(0,499,[1,50]),dtype=torch.int64).to('cuda:0')
        num=graph.num_nodes()-graph.num_nodes()//20
        while graph.num_nodes()>num:
            graph.remove_nodes(torch.randint(0,graph.num_nodes(),[1,1]).reshape(-1))
        return graph
    #边扰动
    elif type==2:
        #随机增删5%的边
        num=graph.num_edges()//20
        for i in range(num):
            graph.remove_edges(torch.randint(0,graph.num_edges(),[1,1]).reshape(-1))
        for i in range(num):
            src=torch.randint(0,graph.num_nodes(),[1,1]).reshape(-1)
            dst=torch.randint(0,graph.num_nodes(),[1,1]).reshape(-1)
            # while(dst==src):
            #     dst=torch.randint(0,graph.num_nodes(),[1,1]).reshape(-1).to('cuda:0')
            if(src!=dst):
                graph.add_edges(src,dst)
        graph=dgl.add_self_loop(graph,etype='_E')
        return graph
    else:
        raise Exception('TYPE ERROR')

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
    def __init__(self, united_dim, apig_dim, fcg_dim, hidden_dim, num_classes):
        super(MultiGraphClassifier, self).__init__()
        self.linear1=nn.Linear(hidden_dim, united_dim)
        self.linear2=nn.Linear(united_dim, hidden_dim)
        
        self.conv_apig1 = dglnn.GraphConv(apig_dim, hidden_dim)
        #self.conv_apig1 = dglnn.GraphConv(apig_dim, hidden_dim)
        self.conv_apig2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv_fcg1 = dglnn.GraphConv(fcg_dim, hidden_dim)
        #self.conv_fcg1 = dglnn.GraphConv(fcg_dim, hidden_dim)
        self.conv_fcg2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        
        self.attention = Attention(hidden_dim)
        self.dropout=nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, apig=None, apig_feat=None, fcg=None, fcg_feat=None):
        if fcg==None:
            #apig_feat=self.linear1(apig_feat)
            
            h_a1 = F.relu(self.conv_apig1(apig, apig_feat))
            a_encode=self.linear1(h_a1)
            a_decode=self.linear2(a_encode)
            h_a2 = F.relu(self.conv_apig2(apig, a_decode))
            
            with apig.local_scope():
                apig.ndata['h'] = h_a2
                apig_embedding = dgl.mean_nodes(apig, 'h').reshape(1,-1).squeeze()
            embedding=self.dropout(apig_embedding)
            output = self.classifier(embedding)
            for val in output:
                if np.isnan(val.cpu().detach().numpy()):
                    print(f"nan occured")
                    exit(0)
            return output
        elif apig==None:
            #fcg_feat=self.linear2(fcg_feat)

            h_f1 = F.relu(self.conv_fcg1(fcg, fcg_feat))
            f_encode=self.linear1(h_f1)
            f_decode=self.linear2(f_encode)
            h_f2 = F.relu(self.conv_fcg2(fcg, f_decode))

            with fcg.local_scope():
                fcg.ndata['h'] = h_f2
                fcg_embedding = dgl.max_nodes(fcg, 'h').reshape(1,-1).squeeze()
            embedding=self.dropout(fcg_embedding)
            output = self.classifier(embedding)
            for val in output:
                if np.isnan(val.cpu().detach().numpy()):
                    print(f"nan occured")
                    exit(0)
            return output
        else:
            # apig_feat=self.linear1(apig_feat)
            # fcg_feat=self.linear2(fcg_feat)
            
            h_a1 = F.relu(self.conv_apig1(apig, apig_feat))
            h_f1 = F.relu(self.conv_fcg1(fcg, fcg_feat))
            
            fusion_ratio=0.1
            a_encode=self.linear1(h_a1)
            f_encode=self.linear1(h_f1)
            a_sum=torch.sum(a_encode,dim=0)
            f_sum=torch.sum(f_encode,dim=0)
            a_encode+=fusion_ratio*f_sum
            f_encode+=fusion_ratio*a_sum
            # for i in range(a_encode.size(0)):
            #     a_encode[i]+=fusion_ratio*f_sum
            # for i in range(f_encode.size(0)):
            #     f_encode[i]+=fusion_ratio*a_sum
            a_decode=self.linear2(a_encode)
            f_decode=self.linear2(f_encode)

            h_a2 = F.relu(self.conv_apig2(apig, a_decode))
            h_f2 = F.relu(self.conv_fcg2(fcg, f_decode)) 
            # h_a2 = F.relu(self.conv_apig2(apig, h_a1))
            # h_f2 = F.relu(self.conv_fcg2(fcg, h_f1))
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

    model = MultiGraphClassifier(united_dim=128, apig_dim=115, fcg_dim=226, hidden_dim=64, num_classes=2)
    model = model.to('cuda')

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses=[]
    # 训练模型
    print("-------------------train begin at ",datetime.now(),"-------------------")
    model.train()
    for epoch in range(100):
        total = 0
        correct = 0
        correct_apig=0
        correct_fcg=0
        TP=0
        FN=0
        FP=0
        TN=0
        for i, (apigs_s, fcgs_s, labels) in enumerate(train_loader):
            #graphs=[fcg_mask(graph,random.randint(0,2)).to('cuda') for graph in graphs]
            apigs=[dgl.add_self_loop(graph).to('cuda') for graph in apigs_s]
            apig_feats=[]
            for apig in apigs:
                apig_feats.append(apig.ndata['feat'].to('cuda'))
            
            fcgs=[dgl.add_self_loop(graph).to('cuda') for graph in fcgs_s]
            fcg_feats=[]
            for fcg in fcgs:
                fcg_feats.append(fcg.ndata['api'].to('cuda'))
            
            labels = labels.to('cuda')
            optimizer.zero_grad()
            #outputs = [model(api_seq, graph, graph_feat) for api_seq, graph,graph_feat in zip(api_seqs, graphs, graph_feats)]
            outputs = [model(apig, apig_feat, fcg, fcg_feat) for apig, apig_feat, fcg, fcg_feat in zip(apigs, apig_feats, fcgs, fcg_feats)]
            outputs=torch.stack(outputs)
            
            outputs_apig = [model(apig=apig, apig_feat=apig_feat) for apig, apig_feat in zip(apigs, apig_feats)]
            outputs_apig=torch.stack(outputs_apig)
            
            outputs_fcg = [model(fcg=fcg, fcg_feat=fcg_feat) for  fcg, fcg_feat in zip(fcgs, fcg_feats)]
            outputs_fcg=torch.stack(outputs_fcg)
            
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算准确率
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            _, predicted_apig = torch.max(outputs_apig.data, 1)
            correct_apig += (predicted_apig == labels).sum().item()
            
            _, predicted_fcg = torch.max(outputs_fcg.data, 1)
            correct_fcg += (predicted_fcg == labels).sum().item()
            
            for i in range(labels.size(0)):
                if predicted[i]==1 and labels[i]==1:
                    TP+=1
                elif predicted[i]==1 and labels[i]==0:
                    FP+=1
                elif predicted[i]==0 and labels[i]==1:
                    FN+=1
                else:
                    TN+=1
        
        accuracy = round(correct / total, 3)
        accuracy_apig=round(correct_apig / total, 3)
        accuracy_fcg=round(correct_fcg / total, 3)
        precision=round(TP/(TP+FP),3)
        recall=round(TP/(TP+FN),3)
        F1score=round(2*(precision*recall)/(precision+recall),3)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}, F1-score:{F1score}, Accuracy_APIG:{accuracy_apig}, Accuracy_FCG:{accuracy_fcg}')
    print("-------------------train end at ",datetime.now(),"-------------------")
    torch.save(model.state_dict(), f'./model/model_20240728_2gcn_modalFusion.pth')