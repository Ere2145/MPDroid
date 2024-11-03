# -*- coding: UTF-8 -*-
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
from pathlib import Path
import os
import numpy as np
#import joblib as J
import networkx as nx
import torch
import os 
import copy
import gc


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
        
    def process_nolabel(self,path):
        print("Process nolabel dataset......................")
        print("load fcgs from ",path)
        fcg_path= os.path.join(path)
        fcg_names=os.listdir(fcg_path)
        cnt=0
        for filename in fcg_names:
            if cnt>3000:
                return
            filepath=os.path.join(fcg_path,filename)
            dg, _ = load_graphs(filepath,[0])
            self.graphs.append(dg[0])
            del dg
            gc.collect()
            print(cnt,filepath)
            cnt+=1
            

        
    def getitem(self, idx):
        return self.graphs[idx]
    
    
    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        """数据集中图的数量"""
        return len(self.graphs)
    
    def __save__(self,name):
        graph_path = os.path.join(self.save_path, name)
        save_graphs(str(graph_path), self.graphs)
        
    def rmv_zerodegree(self):
        """移除0入度节点"""
        print("remove zero-degree node on ",self.graphs[0].device)
        for i in range(len(self.graphs)):
            print('rmv_zerodegree',i)
            #print("before remove 0-degree nodes:",g.num_nodes())
            degrees=self.graphs[i].in_degrees()
            zero_degree_nodes=torch.nonzero(degrees==0).squeeze()
            self.graphs[i].remove_nodes(zero_degree_nodes)
            #print("after remove 0-degree nodes:",g.num_nodes())
    
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
            print('unify_dim',i)
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
                
    
    def addselfloop(self):
        """给0degree节点添加自旋"""
        print("addselfloop on",self.graphs[0].device)
        for i in range(len(self.graphs)):
            print('addselfloop',i)
            self.graphs[i]=dgl.add_self_loop(self.graphs[i],etype='_E')
        
    def __load__(self, name):
        print("start loading data")
        graph_path = os.path.join(self.save_path, name)
        graphs_cpu,_= load_graphs(graph_path)
        for g in graphs_cpu:
            self.graphs.append(g)
        print("finish loading data")

    def __num_tasks__(self):
        """每个图的标签数，即预测任务数。"""
        return 2

    def extract_feat(self):
        """特征处理"""
        """全部特征：
        api(二维向量)
        codesize(一维向量)
        static(一维向量)
        public(一维向量)
        user(二维向量)
        native(一维向量，意义不大)
        entrypoint(一维向量，意义不大)
        external(一维向量)
        """
        #将codesize,static,public与external组合成feat
        print(len(self.graphs))
        for i in range(len(self.graphs)):
            print('extract_feat',i)
            dic=copy.deepcopy(self.graphs[i].ndata)
            #print(type(dic))
            self.graphs[i].ndata.pop('codesize')
            self.graphs[i].ndata.pop('static')
            self.graphs[i].ndata.pop('public')
            self.graphs[i].ndata.pop('user')
            self.graphs[i].ndata.pop('native')
            self.graphs[i].ndata.pop('entrypoint')
            self.graphs[i].ndata.pop('external')
            #print(dic['codesize'],dic['static'],dic['public'],dic['external'])
            feat=torch.zeros(len(dic['codesize']),4)
            for j in range(len(dic['codesize'])):
                feat[j]=torch.Tensor([dic['codesize'][j],dic['static'][j],dic['public'][j],dic['external'][j]])
            #feat=torch.stack([dic['codesize'],dic['static'],dic['public'],dic['external']])
            #print("this is feat:",feat)
            self.graphs[i].ndata['feat']=feat
        #print("check updata",self.graphs[0].ndata)
            
    def rmv_smallgraph(self):
        #l=len(self.graphs)
        print(len(self.graphs))
        for g in self.graphs:
            if(g.num_nodes()<50):
                self.graphs.remove(g)
        print(len(self.graphs))
        
                

if __name__=='__main__':
    gpu_setup()
    data=MyDataset()
    
    data.__load__('pretrain_dataset.bin')
    data.rmv_smallgraph()
    #data.process_nolabel('../../fcg/VirusShare_2013_FCG')
   
    #data.rmv_zerodegree()
    #data.unify_dim()
    #data.addselfloop()
    #data.extract_feat()

    data.__save__('pretrain_dataset_2.bin')
