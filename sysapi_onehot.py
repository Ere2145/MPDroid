import os
import numpy as np

# 准备一个全零的117维向量作为one-hot编码的模板
def create_one_hot(index):
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot

def process_folder(input_folder, output_folder):
    cnt=0
    for filename in os.listdir(input_folder):
        cnt+=1
        print(cnt,filename)
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取特征序列
        feature_sequence = np.load(input_path)

        # 转换为one-hot编码
        one_hot_sequence = np.array([create_one_hot(index - 1) for index in feature_sequence])
        # 保存到输出文件
        np.save(output_path, one_hot_sequence)

if __name__=="__main__":

    # 定义字典总大小
    vocab_size = 115

    # 输入文件夹和输出文件夹
    # input_folder = "./sys_api_vec_total/mal"
    # output_folder = "./sys_api_vec_total/mal_1hot"
    # process_folder(input_folder, output_folder)

    input_folder = "./sys_api_vec_total/mal"
    output_folder = "./sys_api_vec_total/mal_1hot"
    process_folder(input_folder, output_folder)
