import numpy as np
import os
import dgl
import torch
from datetime import datetime

def seq_to_graph_show(src_path):
    arr=np.load(src_path)
    if len(arr)>50000:
        arr=arr[:50000]
    G=dgl.graph(([],[]))
    unique_nodes = np.unique(arr)
    G.add_nodes(np.max(unique_nodes))
    G.ndata['feat']=torch.zeros(np.max(unique_nodes))
    # 遍历数组中的所有元素
    for i in range(len(arr) - 1):
        # 如果节点不存在，添加节点并设置特征
        G.ndata['feat'][arr[i]] += 1.
        # 添加边
        if i < len(arr)-1 and arr[i]!=arr[i+1]:
            G.add_edges(arr[i], arr[i+1])
        if i < len(arr)-2 and arr[i]!=arr[i+2]:
            G.add_edges(arr[i], arr[i+2])
        # 保存图
    feats = G.ndata['feat']
    print(feats)
    non_zero_nodes = torch.nonzero(feats, as_tuple=True)[0]
    sub_G = G.subgraph(non_zero_nodes)
    sub_G.ndata['feat'] = G.ndata['feat'][non_zero_nodes]
    print(len(sub_G.ndata['feat']))

def seq_to_graph(arr=None,savepath=None):
    G=dgl.graph(([],[]))
    unique_nodes = np.unique(arr)
    G.add_nodes(np.max(unique_nodes))
    G.ndata['feat']=torch.zeros(np.max(unique_nodes))
    # 遍历数组中的所有元素
    for i in range(len(arr) - 1):
        # 如果节点不存在，添加节点并设置特征
        G.ndata['feat'][arr[i]] += 1.
        # 添加边
        if i < len(arr)-1 and arr[i]!=arr[i+1]:
            G.add_edges(arr[i], arr[i+1])
        if i < len(arr)-2 and arr[i]!=arr[i+2]:
            G.add_edges(arr[i], arr[i+2])
        # 保存图
    # feats = G.ndata['feat']
    # non_zero_nodes = torch.nonzero(feats, as_tuple=True)[0]
    # sub_G = G.subgraph(non_zero_nodes)
    # sub_G.ndata['feat'] = G.ndata['feat'][non_zero_nodes]
    dgl.save_graphs(savepath, [G])

 
def create_dgl_graph_from_time_series(src_dir, save_dir):
    # 遍历目录下的所有文件
    cnt=0
    start=datetime.now()
    total=0
    for filename in os.listdir(src_dir):
        if filename.endswith('.npy'):
            graph_name = filename.replace('.npy', '.apig')
            save_path=os.path.join(save_dir, graph_name)
            basename,_=os.path.splitext(filename)
            print(cnt,basename,datetime.now())
            cnt+=1            
            if os.path.exists(save_path):
                continue
            # 加载numpy数组
            arr = np.load(os.path.join(src_dir, filename))
            if len(arr)>50000:
                arr=arr[:50000]
            seq_to_graph(arr,save_path)
    end=datetime.now()
    print("from ",start,"to",end)


if __name__=="__main__":
    # 使用函数创建图并保存
    #seq_to_graph_show('./api_src/sys_api_vec_total/bni/air.com.thetourtracker.cntt.csv.txt.npy')
    create_dgl_graph_from_time_series('./api_src/sys_api_vec_total/bni', './api_src/api_graph/bni')
    create_dgl_graph_from_time_series('./api_src/sys_api_vec_total/mal', './api_src/api_graph/mal')
