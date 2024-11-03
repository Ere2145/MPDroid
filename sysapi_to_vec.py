import os
import numpy as np

if __name__=='__main__':
    src_mal_path='./sys_api_seq_total/mal'
    src_bni_path='./sys_api_seq_total/bni'
    dst_mal_path='./sys_api_vec_total/mal'
    dst_bni_path='./sys_api_vec_total/bni'
    
    src_test_path='./sys_api_seq_test'
    dst_test_path='./sys_api_vec_test'
    
    sysapi_set=set()
    
    #存字典
    cnt=0
    for filename in os.listdir(src_bni_path):
        seqfile=os.path.join(src_bni_path,filename)
        with open(seqfile,'r') as f:
            for line in f:
                sysapi=line.strip()
                if sysapi not in {"futex", "clock_gettime"}:
                    sysapi_set.add(sysapi)
    
    for filename in os.listdir(src_mal_path):
        seqfile=os.path.join(src_mal_path,filename)
        with open(seqfile,'r') as f:
            for line in f:
                sysapi=line.strip()
                if sysapi not in {"futex", "clock_gettime"}:
                    sysapi_set.add(sysapi)
                    
    with open('sysapi_words_total.txt','w') as f:
        for word in sysapi_set:
            f.write(word+'\n')
    
    #读字典
    # with open('sysapi_words.txt','r') as f:
    #     for line in f:
    #         sysapi=line.strip()
    #         sysapi_set.add(sysapi)
            
    sysapi_list=list(sysapi_set)
    sysapi_dict={}

    for i,str in enumerate(sysapi_list):
        sysapi_dict[str]=i

    cnt=0
    for filename in os.listdir(src_test_path):
        cnt+=1
        print(cnt,filename)
        seqfile=os.path.join(src_test_path,filename)
        with open(seqfile,'r') as f:
            sysapi_list=[]
            for line in f:
                sysapi=line.strip()
                if sysapi not in {"futex", "clock_gettime"}:
                    sysapi_list.append(sysapi_dict[sysapi])
            #print(sysapi_vec)
            sysapi_vec=np.array(sysapi_list)
            dst_file=os.path.join(dst_test_path,filename)
            np.save(dst_file,sysapi_vec)

    # cnt=0
    # for filename in os.listdir(src_bni_path):
    #     cnt+=1
    #     print(cnt,filename)
    #     seqfile=os.path.join(src_bni_path,filename)
    #     with open(seqfile,'r') as f:
    #         sysapi_list=[]
    #         for line in f:
    #             sysapi=line.strip()
    #             if sysapi not in {"futex", "clock_gettime"}:
    #                 sysapi_list.append(sysapi_dict[sysapi])
    #         #print(sysapi_vec)
    #         sysapi_vec=np.array(sysapi_list)
    #         dst_file=os.path.join(dst_bni_path,filename)
    #         np.save(dst_file,sysapi_vec)
    
    # cnt=0
    # for filename in os.listdir(src_mal_path):
    #     cnt+=1
    #     print(cnt,filename)
    #     seqfile=os.path.join(src_mal_path,filename)
    #     with open(seqfile,'r') as f:
    #         sysapi_list=[]
    #         for line in f:
    #             sysapi=line.strip()
    #             if sysapi not in {"futex", "clock_gettime"}:
    #                 sysapi_list.append(sysapi_dict[sysapi])
    #         #print(sysapi_vec)
    #         sysapi_vec=np.array(sysapi_list)
    #         dst_file=os.path.join(dst_mal_path,filename)
    #         np.save(dst_file,sysapi_vec)