# -*- coding: UTF-8 -*-
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
from dgl.data.utils import load_labels
from dgl.data.utils import save_graphs
from pathlib import Path
#from dgl.data.utils import save_labels
import os
import numpy as np
#import joblib as J
import networkx as nx
import torch
import os 
import copy
import gc
import random


def gpu_setup():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if torch.cuda.is_available():
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

class MyDataset(DGLDataset):
    def __init__(self, raw_dir='/mnt/sda/sh/', force_reload=False, verbose=False, save_dir='../dataset'):
        super(MyDataset, self).__init__(name='',
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose,
                                        save_dir=save_dir)

    def process(self):
        self.graphs=[]
        self.labels=[]
    
    
    def process_label(self,path):
        print("Process labeled dataset......................")
        fcg_path=path
        #mal_path= os.path.join(fcg_path,"mal_fcg")
        mal_path= os.path.join(fcg_path,"VirusShare_2013_FCG")
        bni_path= os.path.join(fcg_path,"bni_fcg")
        
        #将数据处理为图列表和标签列表
        mal_names=os.listdir(mal_path)
        bni_names=os.listdir(bni_path)
        cnt=0
        for filename in mal_names:
            if cnt<500:
                filepath=os.path.join(mal_path,filename)
                dg, _ = load_graphs(filepath,[0])
                g=self.extract_feat(dg[0])
                self.graphs.append(g.to('cuda:0'))
                self.labels.append(1)
                cnt=cnt+1
                print(cnt,filepath)
        cnt=0
        for filename in bni_names:
            if cnt<500:
                filepath=os.path.join(bni_path,filename)
                dg, _ = load_graphs(filepath,[0])
                g=self.extract_feat(dg[0]).to('cuda:0')
                self.graphs.append(g.to('cuda:0'))
                self.labels.append(0)
                cnt=cnt+1
                print(cnt,filepath)
        self.labels=torch.Tensor(self.labels).to('cuda:0')
        gc.collect()
        self.topK()
        self.addselfloop()

        
    def getitem(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """数据集中图的数量"""
        return len(self.graphs)
    
    def __save__(self,name):
        graph_path = os.path.join(self.save_path, name)
        save_graphs(str(graph_path), self.graphs, {'labels': self.labels})
        
    def topK(self):
        print("topK nodes")
        for i,g in enumerate(self.graphs):
            print(i)
            #print("before topK:nodes {} edges {}".format(g.num_nodes(),g.num_edges()))
            #聚合敏感api调用数量为0的节点，保持连通性
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
            aggr_node_edge_src_t=torch.tensor(aggr_node_edge_src_l,dtype=int).to('cuda:0')
            aggr_node_edge_dst_t=torch.tensor(aggr_node_edge_dst_l,dtype=int).to('cuda:0')
            g.add_edges(aggr_node_edge_src_t,aggr_node_edge_dst_t,etype='_E')
            #添加一阶邻居间的边，可能产生环导致死循环，暂时不用
            # for idx in node_to_rmv_l:
            #     frontier=g.sample_neighbors(idx,-1)
            #     neighbor_t=frontier.edges()
            #     edges_to_add_src=[]
            #     edges_to_add_dst=[]
            #     for src in neighbor_t[0]:
            #         src=src.item()
            #         if(src!=idx):
            #             for dst in neighbor_t[1]:
            #                 dst=dst.item()
            #                 if(dst!=idx):
            #                     print("add edge from {} to {}".format(src,dst))
            #                     edges_to_add_src.append(src)
            #                     edges_to_add_dst.append(dst)
            #     edges_to_add_src_t=torch.tensor(edges_to_add_src,dtype=int).to('cuda:0')
            #     edges_to_add_dst_t=torch.tensor(edges_to_add_dst,dtype=int).to('cuda:0')
            #     g.add_edges(edges_to_add_src_t,edges_to_add_dst_t,etype='_E')       
            g.remove_nodes(node_to_rmv)
            #保证新图的联通性
            ## node_num=g.num_nodes()
            #生成边
            # for src in g.nodes():
            #     edges_num=random.randint(node_num//200,node_num//100)
            #     for cnt in range(edges_num):
            #         dst=random.randint(0,node_num)
            #         if(dst!=src):
            #             #print("add edge from {} to {}".format(src,dst))
            #             g.add_edges(src,dst)
            #print("after aggr nonapi:{} nodes,{} edges".format(g.num_nodes(),g.num_edges()))
            """聚合节点"""
            rolls=0
            while(g.num_nodes()>1000):
                #计算节点优先级，考虑敏感api调用数量以及节点出度;对重要节点和不重要的topK均聚合,聚合权重暂时设为邻居数量的倒数
                #print("rolls {},nodes {}".format(rolls,g.num_nodes(),g.num_edges()))
                pri_score=torch.zeros(g.num_nodes())
                out_degrees=g.out_degrees()
                api=torch.sum(g.ndata['api'],dim=1)
                pri_score=api+out_degrees
                k=int(0.1*g.num_nodes())
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
                node_to_rmv=torch.tensor(node_to_rmv,dtype=int).to('cuda:0')
                g.remove_nodes(node_to_rmv)
                rolls+=1
            # """将出入度均为0的节点合成一个孤立节点"""
            # deg=g.in_degrees()+g.out_degrees()
            # print(deg)
            # exit(0)
            # node_to_rmv=[]
            # new_api=torch.zeros(g.ndata['api'][0].shape).to('cuda:0')
            # for i in range(g.num_nodes()):
            #     if deg[i]==0:
            #         node_to_rmv.append(i)
            #         new_api+=g.ndata['api'][i]
            # node_to_rmv=torch.tensor(node_to_rmv,dtype=int).to('cuda:0')
            # g.remove_nodes(node_to_rmv)
            # new_api=torch.unsqueeze(new_api,0)
            # dgl.add_nodes(g,1,{'api':new_api},ntype='_N')
            #print("after topK:nodes {} edges {}".format(g.num_nodes(),g.num_edges()))

            
        
    
    def unify_dim(self):
        """统一特征维度"""
        print("unify_dim on",self.graphs[0].device)
        l=0
        """固定节点数量"""
        l=1000
        # """采用平均长度0填充爆显存"""
        # total_nodes=sum(g.num_nodes() for g in self.graphs)
        # l=total_nodes/len(self.graphs)
        # l=int(l)
        # print("average num_nodes=",l)
        
        
        #"""采用最大长度0填充爆显存"""
        # for g in self.graphs:
        #     l=max(l,len(g.ndata[feat_name]))

        for i in range(len(self.graphs)):
            print(i)
            numnodes = self.graphs[i].num_nodes()
            if numnodes < l:
                # 补充节点
                num_nodes_to_add = l - numnodes
                new_nodes = torch.arange(numnodes, numnodes + num_nodes_to_add).to('cuda:0')
                self.graphs[i].add_nodes(num_nodes_to_add)
                self.graphs[i].add_edges(new_nodes, new_nodes)  # 添加自旋边
            elif numnodes > l:
                # 删除编号大于 l 的节点
                nodes_to_remove = torch.arange(l, numnodes).to('cuda:0')
                # 删除包含这些节点的边
                # #self.graphs[i].remove_edges(self.graphs[i].edge_ids(nodes_to_remove, torch.arange(numnodes).to('cuda:0')))
                self.graphs[i].remove_nodes(nodes_to_remove)
                
            
    
    def fix_label(self):
        """标签生成"""
        print("fix_label")
        newlabels=torch.empty(len(self.labels),2).to('cuda:0')
        for i in range(len(self.labels)):
            print(i)
            if(self.labels[i]==1):
                newlb=torch.Tensor([1,0]).to('cuda:0')
                newlabels[i]=newlb
            else:
                newlb=torch.Tensor([0,1]).to('cuda:0')
                newlabels[i]=newlb
        print(type(newlabels))
        print(newlabels)
        self.labels=newlabels
        print("fix_label on",self.labels.device)
    
    def addselfloop(self):
        """给0degree节点添加自旋"""
        gc.collect()
        print("addselfloop on",self.graphs[0].device)
        for i in range(len(self.graphs)):
            print(i)
            self.graphs[i]=dgl.add_self_loop(self.graphs[i],etype='_E')
        
    def __load__(self, name):
        print("start loading data")
        graph_path = os.path.join(self.save_path, name)
        #graphs, label_dict = load_graphs(graph_path)
        graphs_cpu, label_dict_cpu = load_graphs(graph_path)
        for g in graphs_cpu:
            graph_gpu=g.to('cuda:0')
            self.graphs.append(graph_gpu)
            # print(graphs_cpu)
            # print(label_dict_cpu)
        self.labels=label_dict_cpu['labels'].to('cuda:0')
        
        # print(self.graphs[0].device)
        # print(self.labels.device)
        print("finish loading data")

    def __num_tasks__(self):
        """每个图的标签数，即预测任务数。"""
        return 2

    def extract_feat(self,graph):
        """特征处理"""
        """全部特征：
        api
        codesize
        static
        public
        user
        native
        entrypoint
        external
        """
        graph.ndata.pop('codesize')
        graph.ndata.pop('static')
        graph.ndata.pop('public')
        graph.ndata.pop('user')
        graph.ndata.pop('native')
        graph.ndata.pop('entrypoint')
        graph.ndata.pop('external')
        return graph
        
        #将codesize,static,public与external组合成feat
        #gc.collect()
        #print(len(self.graphs))
        #for i in range(len(self.graphs)):
            #dic=copy.deepcopy(self.graphs[i].ndata)
            #print(type(dic))
            #self.graphs[i].ndata.pop('api')
            # self.graphs[i].ndata.pop('codesize')
            # self.graphs[i].ndata.pop('static')
            # self.graphs[i].ndata.pop('public')
            # self.graphs[i].ndata.pop('user')
            # self.graphs[i].ndata.pop('native')
            # self.graphs[i].ndata.pop('entrypoint')
            # self.graphs[i].ndata.pop('external')
            #print(dic['codesize'],dic['static'],dic['public'],dic['external'])
            #feat=torch.zeros(len(dic['codesize']),4).to('cuda:0')
            # for j in range(len(dic['codesize'])):
            #     feat[j]=torch.Tensor([dic['codesize'][j],dic['static'][j],dic['public'][j],dic['external'][j]])
            #feat=torch.stack([dic['codesize'],dic['static'],dic['public'],dic['external']])
            #print("this is feat:",feat)
            #self.graphs[i].ndata['feat']=feat
        #print("check updata",self.graphs[0].ndata)
            

if __name__=='__main__':
    gpu_setup()
    gc.enable()
    
    data=MyDataset()
    #data.__load__('FCG_CIC&drebin.bin')
    #no_label
    data.process_label("../../fcg")
    
    #data.__save__()
    #data.__load__('FCG_CIC&drebin3.bin')
    #data.__load__('smallDataset4.bin')
    
    
    #data.rmv_zerodegree()
    #data.fix_label()
    #data.unify_dim()
    # data.addselfloop()
    # data.extract_feat()
    
    
    #data.__save__('smallDataset5.bin')
    data.__save__('FCG_CIC&drebin_api.bin')
    #data.__save__('pretrain_dataset.bin')
    
    #print(data.getitem(0)[0].ndata['feat'].shape)
