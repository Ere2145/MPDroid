import pandas as pd
import chardet
import re
import os


def extract_syscalls(series):
    syscalls = []
    for line in series:
        match = re.match(r'(\w+)\(', line)
        if match:
            syscalls.append(match.group(1))
    return syscalls

if __name__=='__main__':
    
    rawdata = open('../apks/src_dir/bni/Dynamic/Strace/air.com.adobe.connectpro.csv', 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    
    src_path='../apks/src_dir/test/Dynamic/Strace'
    dst_path='./sys_api_seq_test'
    
    cnt=0
    for filename in os.listdir(src_path):
        cnt+=1
        csvfile=os.path.join(src_path,filename)
        # 读取CSV文件
        df = pd.read_csv(csvfile ,encoding=encoding,usecols=range(3),header=None)

        # 获取行数和列数
        # rows, cols = df.shape
        # print(f'行数: {rows}, 列数: {cols}')

        # # 获取列关键字
        # column_names = df.columns.tolist()
        # print(f'列关键字: {column_names}')
        
        # 获取第三列的数据
        syscalls_column = df.iloc[:, 2]
        syscalls = extract_syscalls(syscalls_column)
        # print(syscalls)

        
        # 将数据写入到TXT文件中
        dstfile=os.path.join(dst_path,filename+'.txt')
        with open(dstfile, 'w') as f:
            for item in syscalls:
                f.write("%s\n" % item)
        
        print(cnt,filename)
