#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[6]:


class Affine():
    '''
    一个带有偏置的全链接层，偏置默认是零
    '''
    def __init__(self,W,b=0):
        try:
            import numpy as np
        except:
            print('initialing failed,please install numpy first')
            return
        self.params = {}
        self.params['W'] = W
        self.params['b'] = b
    def forward(self,X):
        return np.dot(X,self.params['W'])+self.params['b']


# In[7]:


#实现一个简单的word2vec前向推理过程
if __name__ == '__main__':
    you = [[1,0,0,0,0,0,0]]
    hello = [[0,0,1,0,0,0,0]]

    W_in = np.random.randn(7,3)
    W_out = np.random.randn(3,7)

    layer_in = Affine(W_in)
    layer_out = Affine(W_out)

    predicts = layer_out.forward(0.5*(layer_in.forward(you)+layer_in.forward(hello)))

    print(predicts.round(2))


# In[8]:


def creat_CBOW_train_data(input_string,window_size=1,out_format = 'one_hot'):
    '''
    实现将一段输入文本，根据窗口大小，以CBOW模型，转换成包含feat和target两部分的train data
    输出的格式可以选择'one-hot'和'id'
    string to corpus,corpus to train and target data, train and target data to one_hot
    '''
    try:
        from chapter2 import s2c
    except:
        print('file addr error,please put current file and chapter2.ipynb in a same file')

    corpus = s2c(input_string) #s2c返回一个字典，包含三个字字典，key分别是corpus，word_to_id,id_to_word
    word_to_id = corpus['word_to_id']
    id_to_word = corpus['id_to_word']
    corpus = corpus['corpus']
    
    corpus_len = len(corpus)
    word_to_id_len = len(word_to_id.items())
    
    if out_format == 'one_hot':
        
        train_feat = np.zeros((int(corpus_len-2*window_size),int(2*window_size),int(word_to_id_len)))
        train_target = np.zeros((int(corpus_len-2*window_size),int(word_to_id_len)))
        
        for n,data_id in enumerate(corpus):                           #对于每一个语料库中的单词进行循环

            if n-window_size >= 0 and n+window_size <= corpus_len-1:  #控制循环不超过语料库的边界

                target_data = np.zeros(word_to_id_len)                  #生成train_target数据
                target_data[data_id] =1
                train_target[n-window_size] = target_data

                for distance in range(1,window_size+1):                 #生成目标左边和右边的train_feat，一个train_feat包含多个数据,一个train_feat包含的数据数量是窗口大小的两倍

                    left_data_index = n-distance
                    right_data_index = n+distance

                    left_data_id = corpus[left_data_index]
                    right_data_id = corpus[right_data_index]

                    left_data = np.zeros(word_to_id_len)
                    left_data[left_data_id] = 1

                    right_data = np.zeros(word_to_id_len)
                    right_data[right_data_id] =1

                    train_feat[n-window_size,window_size-distance] = left_data
                    train_feat[n-window_size,window_size+distance-1] = right_data

        return train_feat,train_target
    
    if out_format == 'id':
        
        train_feat = np.zeros((int(corpus_len-2*window_size),int(2*window_size)))
        train_target = np.zeros(int(corpus_len-2*window_size))
        
        for n,data_id in enumerate(corpus):                           #对于每一个语料库中的单词进行循环

            if n-window_size >= 0 and n+window_size <= corpus_len-1:  #控制循环不超过语料库的边界

                train_target[n-window_size] = int(data_id)            #生成train_target数据

                for distance in range(1,window_size+1):                 #生成目标左边和右边的train_feat，一个train_feat包含多个数据,一个train_feat包含的数据数量是窗口大小的两倍

                    left_data_index = n-distance
                    right_data_index = n+distance

                    left_data_id = corpus[left_data_index]
                    right_data_id = corpus[right_data_index]

                    train_feat[n-window_size,window_size-distance] = left_data_id
                    train_feat[n-window_size,window_size+distance-1] = right_data_id
        
        return train_feat,train_target


# In[11]:


#测试creat_train_data()函数

if __name__ == '__main__':
    input_string = 'you say goodbye and i say hello.'
    train_feat,train_target = creat_CBOW_train_data(input_string,window_size=2,out_format='one_hot')
    print('train_feat:\n',train_feat,'\n')
    print('train_target:\n',train_target)


# In[ ]:




