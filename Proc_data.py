import re

import numpy,os
from keras.preprocessing.sequence import pad_sequences
import pickle,sys
from gensim.models import KeyedVectors
import numpy as np
curPath = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print('curPath',curPath)
sys.path.append(curPath)
import matplotlib.pyplot as plt

class process_data(object):

    num_words = 150000

    def __init__(self):
        self.cn_model = self.load_dict()
        self.embedding_dim=300

    @classmethod
    def change_num_words(cls,Train='KEEP'):
        cls.num_words = 50000
        return cls.num_words

    def comp_maxlen(self,data):
        # 统计每个评价的长度，并可视化来确定评价的最大序列长度
        len_lenth = [len(sent) for sent in data]

        len_lenth = np.array(len_lenth)
        print('原始训练语料的训练长度统计：', '最长语料', np.max(len_lenth), '最短语料', np.min(len_lenth), '语料长度均值', np.mean(len_lenth))

        # 可视化评价的长度统计
        plt.hist(len_lenth, bins=100)
        plt.xlabel('length of words')
        plt.ylabel('number of tokens')
        plt.title('Distribution of Length of Tokens')
        plt.show()

        # 计算出序列'合适'的最大长度,一般采取均值+2倍标准差
        maxlen = np.mean(len_lenth) + 2 * np.std(len_lenth)
        # 注意：统一的序列长度一定要为int
        maxlen = int(maxlen)
        print('统一的序列长度', maxlen)

        # 统计小于该最大长度的序列所占比例，检测是否覆盖超过95%的序列
        seq_por = np.sum(len_lenth < maxlen) / len(len_lenth)
        print('统一的序列长度覆盖的所有序列比例', seq_por)

        # with open('model/config.pkl', 'wb') as outp:
        #     pickle.dump((vocab, chunk_tags), outp)
        return maxlen

    @staticmethod
    def load_dict():
        # 导入预训练语料模型
        import pickle
        #yl_pickle_path = 'Pickle/cn_model.pickle'
        yl_pickle_path = os.path.dirname(curPath)+'/TOOLS/wiki_matrix/cn_model.pickle'

        with open(yl_pickle_path, 'rb') as fr:
            cn_model = pickle.load(fr)
            fr.close()

        # print('维基百科辞典读取完成……………………')
        return cn_model


    def load_data(self):

        train = self.split_data(open(curPath+'/MODEL/Ner_keras/Data/bd_mark_train.txt', 'rb'))
        test = self.split_data(open(curPath+'/MODEL/Ner_keras/Data/bd_mark_test.txt', 'rb'))

        maxlen=self.comp_maxlen(train)
        cn_model=self.load_dict()

        print('词汇表长度为：',self.num_words)

        entity_tags = ['O', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', "B_ORG", "I_ORG",'B_TIME','I_TIME']

        train_x,train_y = self.data_pad(train,cn_model,entity_tags,maxlen)
        test_x,test_y = self.data_pad(test,cn_model,entity_tags,maxlen)

        embedding_matrics=self.ini_embed(cn_model)

        with open(curPath+'/MODEL/Ner_keras/Pickle/config.pkl', 'wb') as outp:
            pickle.dump((self.num_words,entity_tags,self.embedding_dim,embedding_matrics,maxlen), outp)

        return (train_x, train_y), (test_x, test_y), (self.num_words, entity_tags,self.embedding_dim,embedding_matrics,maxlen)


    def ini_embed(self,cn_model):
        # 初始化embedding_matrix（用预训练语料建立要用的embedding_matrix）
        embedding_matrics = np.zeros((self.num_words, self.embedding_dim))
        # 将语言模型中前5万的数值对应输入矩阵中
        for i in range(self.num_words):
            # cn_model.index2word[i]为语言模型中索引i对应的字词
            # cn_model[cn_model.index2word[i]]则为该字词对应的300维度的数值
            embedding_matrics[i, :] = cn_model[cn_model.index2word[i]]
        embedding_matrics = embedding_matrics.astype('float32')

        return embedding_matrics


    def split_data(self,fh):

        string = fh.read().decode('utf-8')
        # 提取实体命名原始数据(原始数据以\n\n分句,句子以\n分词)
        data = [[row.split() for row in sample.split('\n')] for
                sample in string.strip().split('\n' + '\n')]
        fh.close()
        return data

    def data_pad(self,data,cn_model, entity_tags, maxlen=None):

        x = []
        for s in data:
            word2idx=[]
            for w in s:
                try:
                    word2idx.append(cn_model.vocab[w[0]].index)
                except:
                    word2idx.append(0)
            x.append(word2idx)
        print('数据类型是', type(x))
        print('数量是', len(x))

        y_entity = []
        for s in data:
            y = []
            for w in s:
                try:
                    if '2500' in w[1]:
                        w[1] = re.sub('2500','ORG',w[1])
                    elif '2600' in w[1]:
                        w[1] = re.sub('2600', 'ORG', w[1])
                    y.append(entity_tags.index(w[1]))
                except IndexError as e:
                    print(e)
                    print(w)
                    y.append(int(0))
            y_entity.append(y)
        # y_entity = [[entity_tags.index(w[1]) for w in s] for s in data]
        x = pad_sequences(x, maxlen,padding='pre',truncating='pre')

        y_entity = pad_sequences(y_entity, maxlen, value=-1)

        y_entity = numpy.expand_dims(y_entity, 2)
        return x, y_entity

    def pred_pad(self,data):
        x = []
        for w in data:
            try:
                x.append(self.cn_model.vocab[w[0]].index)
            except:
                x.append(0)

        length = len(x)
        # print([x])
        x = pad_sequences([x], maxlen=5154,padding='pre',truncating='pre')
        x[x >= 150000] = 0
        # print(x)
        return x, length
