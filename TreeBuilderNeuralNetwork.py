# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append('../')

import copy
import json
from Doc.Document import Document
import tensorflow as tf
import gensim
import random
import numpy as np
import sklearn.metrics as skmetrics
import jieba
from itertools import chain
import os
import re
import cPickle


__metaclass__ = type

class BuildTrainDataset():
    '''
    build_tree():把 json 文件转成 configurations表的格式，每个文件存成  *.pkl
    build_dataset():把每个 *.pkl 文件存成数据集，train_test_dataset_initial.txt
    build():调用以上函数，并生成 train_dataset.txt, val_dataset.txt,test_dataset.txt

    '''

    def __init__(self,json_path,savedata_path,wordembedding_dir):
        self.json_path=json_path
        self.savedata_path=savedata_path
        self.wordembedding_dir=wordembedding_dir

        if not os.path.isdir(json_path):
            raise Exception('Can\'t find json_path!')
        if not os.path.exists(wordembedding_dir):
            raise Exception('Can\'t find wordembedding file')
        if not os.path.isdir(savedata_path):
            os.makedirs(savedata_path)
            print('the dir is made for saving datasets:',savedata_path)


    def __load_embedding_vocab(self,path):
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        w2v_vocab = word_vectors.vocab.keys()
        w2v_vocab_dict = dict(zip(w2v_vocab, w2v_vocab))
        return w2v_vocab_dict

    def build_tree(self,path):
        '''
        做成  Stack ParagraphList   Action       Relation 格式
            <root>  P0~PN        shift/reduce    Pn --> Pm/[]
        '''
        # if False:
        # if not os.path.exists(os.path.join(self.savedata_path,'tree_configurations.pkl')):
        name_list = os.listdir(path)
        name_list=[i for i in name_list if '.json' in i]
        output = []
        for i in name_list:
            if os.path.exists(os.path.join(self.savedata_path, i.replace('.json', '.pkl'))):
                print(i.replace('.json', '.pkl'),'exists.')
                continue
            print(i)
            states = []
            with open(path + i, 'r') as f:
                js = f.read()
                data = json.loads(js)
                depths=data['depths']
                contents=data['document']['contents']
                data1 = []
                for j in range(len(depths)):
                    if 'cells' not in contents[j].keys():
                        data1.append({'depth':depths[j],'ind':j,'text':contents[j]['text']['text'],'tag':contents[j]['tag'],'other_features':{'number_list':contents[j]['number_list'],'style':contents[j]['style'],'linguistic_feature':contents[j]['text']['linguistic_feature']}})
                    else:
                        data1.append({'depth':depths[j],'ind':j,'text':json.dumps(contents[j][u'cells']),
                                      'tag':contents[j]['tag']})
                data=copy.deepcopy(data1)
                Stack = ['<root>']
                ParagraphList = copy.deepcopy(data)
                Relation = []
                Action = 'shift'

                state = [Stack, ParagraphList, Action, Relation]  ##configuration
                states.append(copy.deepcopy(state))
                while True:

                    # if len(Stack) == 3 and ParagraphList == []:
                    #     print('')  ##debug

                    ##根据前一行的Action来执行当前行操作
                    if Action == 'shift':
                        Stack.append(ParagraphList.pop())
                    elif Action == 'reduce':
                        if len(Stack) > 2:
                            Stack.pop(-2)
                        else:
                            Stack.pop(-1)

                    ##确定当前行的Action and Relation
                    if Stack == ['<root>'] and ParagraphList != []:
                        Action = 'shift'
                        Relation = []
                    elif Stack != ['<root>'] and ParagraphList == []:
                        Action = 'reduce'
                        if len(Stack) > 2:
                            Relation = [Stack[-1], '-->', Stack[-2]]
                        else:
                            Relation = ['<root>', '-->', Stack[-1]]
                    elif Stack != ['<root>'] and ParagraphList != []:
                        # 判断栈顶两个的关系
                        if len(Stack) == 2:
                            if Stack[-1]['depth'] == 1:
                                Action = 'reduce'
                                Relation = ['<root>', '-->', Stack[-1]]
                            else:
                                Action = 'shift'
                                Relation = []
                        else:
                            top1 = Stack[-1]
                            top2 = Stack[-2]
                            if top1['depth'] - top2['depth'] == -1:
                                Action = 'reduce'
                                Relation = [top1, '-->', top2]
                            else:
                                Action = 'shift'
                                Relation = []
                    elif Stack == ['<root>'] and ParagraphList == []:
                        Action = 'Done'
                        Relation = []

                    state = [Stack, ParagraphList, Action, Relation]
                    states.append(copy.deepcopy(state))

                    if Action == 'Done':
                        break

                # output.append(copy.deepcopy(states))
                cPickle.dump(states,open(os.path.join(self.savedata_path, i.replace('.json','.pkl')),'wb'))
            # cPickle.dump(output, open(os.path.join(self.savedata_path,'tree_configurations.pkl'), 'wb'))
        # else:
        #     output = cPickle.load(open(os.path.join(self.savedata_path,'tree_configurations.pkl'), 'rb'))
        #     a=json.dumps(output)
        #     print(os.path.join(self.savedata_path,'tree_configurations.pkl'),'is exist.\nif you want to updata,delete it')
        # return output

    def build_dataset(self):
        '''
        data_configuration  数据格式：
        [Stack,Paragraph List,Action,Relation]

        训练数据使用
        Stack里面栈顶的3段，每段首尾5个词（不足补0）  3×（5+5）*50=1500
        Paragraph List里面的后3段，每段首尾5个词（不足补0） 3×（5+5）*50=1500
        Relation里面当前的 前一个head-->dependent 这2段的每段的首尾5个词（不足补0）   2×（5+5）*50=1000

        新增段落空间位置特征：   is_bold: True/False  -->  1/0                                  1维
                            pattern: 34(pattern.txt)+1(other)  -->  onehot [0,1,0,...,0]   35维
                            font_size: 22.08~88.08  -->  除以100, 0~1                       1维
                            tag:paragraph/table  -->   0/1                                 1维
                            jc:left/center -->   0/1                                       1维
                            indent: 0~64   -->   onehot [0,1,0,...,0]                      65维
                            number: 0~39  -->  除以40 , 0~1                                 1维
                                                                                        合计 =105维  使用4行 105*4=420维

        因此，神经网络模型的输入为1500+1500+1000+105*4=4000+420
        隐藏层节点数 为128
        :param :
        :return:
        '''

        embedding_vocab_dict = self.__load_embedding_vocab(path=self.wordembedding_dir)

        data = []

        name_list = os.listdir(self.savedata_path)
        name_list = [i for i in name_list if '.pkl' in i]

        for name in name_list:
            print(name)
            d=cPickle.load(open(self.savedata_path+'/'+name,'rb'))
            for i in range(0, len(d) - 1):
                tmpd = d[i]
                if tmpd[2] == 'reduce':
                    label_i = 1
                else:  ## 'shift'
                    label_i = 0

                stack_d = tmpd[0][-3:]
                parag_list_d = tmpd[1][-3:]

                s = ['<null>'] * (3 - len(stack_d)) + stack_d  # Stack
                p = parag_list_d + ['<null>'] * (3 - len(parag_list_d))  ##Paragraph List
                for e in range(len(s)):
                    if type(s[e]) == type({}): s[e] = s[e]['text']
                for e in range(len(p)):
                    if type(p[e]) == type({}): p[e] = p[e]['text']

                if i == 0:
                    r = ['<null>', '<null>']
                else:
                    r = d[i - 1][3]
                    if r == []:
                        r = ['<null>', '<null>']
                    else:
                        r = [r[0], r[-1]]  ##Relation
                for e in range(len(r)):
                    if type(r[e]) == type({}): r[e] = r[e]['text']
                ##s p r 每个取首尾5个词
                l = s + p + r
                ##判断l是否遇到<root> <null>
                data_d = []
                for a in l:
                    if a != '<root>' and a != '<null>':
                        sentence_list = [jj for jj in ' '.join(jieba.cut(a, cut_all=False)).split() if
                                         jj in embedding_vocab_dict]
                        if len(sentence_list) >= 5:
                            data_d.append(sentence_list[:5] + sentence_list[-5:])
                        else:
                            data_d.append(
                                sentence_list + ['</s>'] * (5 - len(sentence_list)) * 2 + sentence_list)  ##不足补零
                    elif a == '<root>':
                        data_d.append(['<root>'] * 10)
                    elif a == '<null>':
                        data_d.append(['</s>'] * 10)  ##前五个，后五个

                ### 在data_d中添加手动的空间结构特征
                if r==['<null>', '<null>']:
                    hf=self.__human_feature(tmpd=tmpd,previous_relation=None)
                else:
                    hf=self.__human_feature(tmpd=tmpd,previous_relation=d[i - 1][3])

                data.append([list(chain.from_iterable(data_d+[hf])), label_i])
        with open(self.savedata_path+'/train_test_dataset.txt', 'w') as f:
            for d in data:
                f.write((u' '.join(d[0]) + ' ##' + str(d[1]).encode('utf-8') + '\n').encode('utf-8'))
            print('dataset done.')
        return data


    def __human_feature(self,tmpd,previous_relation):
        '''
            is_bold: True/False  -->  1/0                                  1维
            pattern: 34(pattern.txt)+1(other)  -->  onehot [0,1,0,...,0]   35维
            font_size: 22.08~88.08  -->  除以100, 0~1                       1维
            tag:paragraph/table  -->   0/1                                 1维
            jc:left/center -->   0/1                                       1维
            indent: 0~64   -->   onehot [0,1,0,...,0]                      65维
            number: 0~39  -->  除以40 , 0~1                                 1维
                                                                        合计 =105维  使用4行 105*4=420维
        '''
        pattrens='''pattern:第[零〇一二三四五六七八九十百]+部分
                pattern:（[1-9][0-9]?）
                pattern:第[零〇一二三四五六七八九十百]+条
                pattern:[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?
                pattern:([1-9][0-9]?)
                pattern:[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?
                pattern:[a-z]
                pattern:[1-9][0-9]?．
                pattern:[1-9][0-9]?）
                pattern:附件[零〇一二三四五六七八九十百]+：
                pattern:(XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)+
                pattern:（[零〇一二三四五六七八九十百]+）
                pattern:[1-9][0-9]?.[1-9][0-9]?
                pattern:([零〇一二三四五六七八九十百]+)
                pattern:[零〇一二三四五六七八九十百]+、
                pattern:[-]+
                pattern:(xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)+．
                pattern:[A-Z]
                pattern:[1-9][0-9]?
                pattern:[1-9][0-9]?.
                pattern:[①-⑳]+
                pattern:[1-9][0-9]?)
                pattern:(XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)+：
                pattern:第[1-9][0-9]?章
                pattern:[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?
                pattern:[零〇一二三四五六七八九十百]+
                pattern:(xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)+
                pattern:[A-Z]：
                pattern:第[零〇一二三四五六七八九十百]+章
                pattern:[a-z]．
                pattern:
                pattern:注[1-9][0-9]?：
                pattern:附件[零〇一二三四五六七八九十百]+
                pattern:[1-9][0-9]?、'''.replace('pattern:','').replace(' ','').split('\n')

        try:
            stack_last=[tmpd[0][-1]]
            if tmpd[1]:
                para_list_last=[tmpd[1][-1]]
            else:##队列为空时
                para_list_last=['<null>']

            if previous_relation:
                l=stack_last+para_list_last+[previous_relation[0]]+[previous_relation[-1]]
            else:
                l = stack_last + para_list_last+['<null>', '<null>']

            whole_hf=[]
            for i in l:
                if i=='<null>' or i=='<root>':
                    # hf=['<NoneHF>']*12
                    hf = ['0'] * 105
                elif i['tag']=='table':
                    # hf=['tag:table']+['<NoneHF>']*11
                    hf = ['0'] * 105
                    hf[47]='1'
                else:
                    hf=[]
                    other_features=i['other_features']
                    if other_features['style']['is_bold']==True:
                        hf.append('1')
                    else:hf.append('0')

                    if other_features['number_list'][u'pattern'] in pattrens:
                        tmp=['0']*35
                        tmp[pattrens.index(other_features[u'number_list'][u'pattern'])]='1'
                        hf+=tmp
                    else:
                        tmp=['0']*35
                        tmp[-1]='1'
                        hf+=tmp

                    hf.append(str(float(other_features['style']['font_size'])/100.0))

                    hf.append('0')  ## tag : paragraph

                    if other_features['style']['jc']==u'left':
                        hf.append('0')
                    else:
                        hf.append('1')

                    indent=int(other_features['style']['indent'])
                    tmp = ['0'] * 65
                    if indent<=65:
                        tmp[indent]='1'
                    hf+=tmp

                    hf.append(str(float(other_features['number_list']['number'])/40.0))
                    # hf=['tag:'+str(i['tag']),
                    #     'dp_tree_paths:'+str(other_features['linguistic_feature']['dp_tree_paths']),
                    #     'named_entities:'+str(other_features['linguistic_feature']['named_entities']),
                    #     'pos_tags:'+str(other_features['linguistic_feature']['pos_tags']),
                    #     'segments:'+str(other_features['linguistic_feature']['segments']),
                    #     'number:'+str(other_features['number_list']['number']),
                    #     'pattern:'+other_features['number_list']['pattern'],
                    #     'font:'+str(other_features['style']['font']),
                    #     'font_size:' + str(other_features['style']['font_size']),
                    #     'indent:'+str(other_features['style']['indent']),
                    #     'is_bold:'+str(other_features['style']['is_bold']),
                    #     'jc:'+str(other_features['style']['jc'])]
                whole_hf += hf

        except:
            print('')

        return whole_hf

    def tongji(self):
        with open(self.savedata_path + '/train_test_dataset_initial.txt', 'r') as f:
            txt_list=f.read().strip().split('\n')
            tongji_list=[]
            for i in range(len(txt_list)):
                tongji_list+=txt_list[i].split()[80:]
            tongji_list=list(set(tongji_list))

        with open(self.savedata_path + '/pattern.txt', 'w') as ff:  ##保存 pattern
            for e in [i for i in tongji_list if 'pattern:' in i]:
                ff.write(e + '\n')

            print('')


    def bulid(self):
        print('build files steps:\n *.json --> *.pkl --> train_test_dataset.txt --> (train_dataset.txt, val_dataset.txt, test_dataset.txt)')
        # self.tongji()
        self.build_tree(path=self.json_path)
        print('*.pkl s done')
        dataset = self.build_dataset()
        print('rain_test_dataset.txt done')
        random.seed(10)  ##设置种子使得每次抽样结果相同
        random.shuffle(dataset)
        random.shuffle(dataset)
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
        test_dataset = dataset[int(len(dataset) * 0.9):]
        # with open(os.path.join(self.savedata_path,'train_test_dataset.txt'), 'r') as f:
        #     data_txt_list = f.read().strip().split('\n')
        #     random.shuffle(data_txt_list)
        #     random.shuffle(data_txt_list)
        # train_dataset = data_txt_list[:int(len(data_txt_list) * 0.8)]
        # val_dataset = data_txt_list[int(len(data_txt_list) * 0.8):int(len(data_txt_list) * 0.9)]
        # test_dataset = data_txt_list[int(len(data_txt_list) * 0.9):]

        with open(self.savedata_path+'/train_dataset.txt', 'w') as f:
            for d in train_dataset:
                f.write((u' '.join(d[0]) + ' ##' + str(d[1]).encode('utf-8') + '\n').encode('utf-8'))
            print('train_dataset done.')
        with open(self.savedata_path+'/val_dataset.txt', 'w') as f:
            for d in val_dataset:
                f.write((u' '.join(d[0]) + ' ##' + str(d[1]).encode('utf-8') + '\n').encode('utf-8'))
            print('val_dataset done.')
        with open(self.savedata_path+'/test_dataset.txt', 'w') as f:
            for d in test_dataset:
                f.write((u' '.join(d[0]) + ' ##' + str(d[1]).encode('utf-8') + '\n').encode('utf-8'))
            print('test_dataset done.')


class DocumentCorrection():
    def __init__(self):
        pass
    def delete_catalog(self,doc_contents):
        '''
        delete the redundant catalog
        :return:
        '''
        return doc_contents

    @staticmethod
    def merge_titles(doc_contents, span=10):
        '''
        ## merge titles when an document has more than one title
        :param doc_contents: a list, include all sentence
               span: if span equal -1, go through all doc_contents, else go through doc_contents[0:span]
        :return: doc_contents after merging titles
        '''
        tag_has_title=True

        def is_chapter(doc_content):
            special_symbol_begining = u'① ② ③ ➀ ➁ ➂ ➊ ➋ ➌     ⑴ ⑵ ⑶ ㈠ ㈡ ㈢ ㊀ ㊁ ㊂ 􀁮 􀁙 􀁚 ' \
                                      u'u v w 􀁘 􀁙 􀁚 􀁛 􀁜 􀁝 􀁞 􀁟 􀁠 􀁡        ● ○ • ◌ ◆ ◇ ♢ ♦ ❖ ■ □ ' \
                                      u'➢ ➣ ➤ →          '.split()
            # chapter1_begining=['第一部分','第一章','第一条','第1张']
            chapter1_begining = re.findall(u'(第[一1 ]+[章节条])|(第[一1 ]+部分)|(^1\.)|(^1 )|(^第[一1 ]+.*)',
                                           doc_content.decode('utf-8'))
            chapter1_words = re.findall(u'(.*[与和]我们[订定]立的合同)|(.*[与和]我们的合同)|(.*合同构成)|(.*拥有的重要权益)|(.*关于本保险合同)',
                                        doc_content.decode('utf-8'))
            chapter_special_symbol = [doc_content for i in special_symbol_begining if i in doc_content.decode('utf-8')]
            return chapter1_begining + chapter1_words + chapter_special_symbol

        chaper1_index = 0
        if span >= len(doc_contents):
            return doc_contents,tag_has_title
        if span == -1:
            span = len(doc_contents)
        for i in range(span):
            if len(is_chapter(doc_contents[i])) > 0:  # find chapter1 begining position
                chaper1_index = i
                if i==0: tag_has_title=False
                break
        if chaper1_index == 0 or chaper1_index == 1:
            return doc_contents,tag_has_title
        else:
            titles = ['<title_enter>'.join(doc_contents[:chaper1_index])]
            contents = doc_contents[chaper1_index:]
            return titles + contents,tag_has_title

    def adjust_enter(self,doc_contents):
        return doc_contents

    def do_correction(self,doc_contents):
        doc_contents1=self.delete_catalog(doc_contents) ## 删除多余目录
        doc_contents2=self.adjust_enter(doc_contents1) ## 调整错误的换行
        doc_contents3,_=DocumentCorrection.merge_titles(doc_contents2) ## 合并多个标题
        return doc_contents3


class TreeBuilderNeuralNetwork(DocumentCorrection):
    """
    Build tree from a structured document by the method of neural network. Configuration contains of Stack, Paragraph
    List(buffer) and Relation. Classifier has two actions, shift and reduce. If shift is selected by Classifier, we push
    a paragraph to Stack from Paragraph List. If reduce is selected, we delete the second paragraph of Stack, and then
    we make an arc of this two paragraphs and save it in Relation. This algorithm is similar to transition-based
    dependency parsing.
    """
    def __init__(self,restore_model_path='Tree_model_saved/',log_dir='./log/',w2v_path='zhwiki_finance_simple.sg_50d.word2vec'):
        #super(TreeBuilderNeuralNetwork, self).__init__() ##or DocumentCorrection.__init__(self) ##不继承类DocumentCorrection的__init__
        self.num_input_units=4000+420   ##4000 is w2v feature, 420 is human feature
        self.num_hiden_layer=1
        self.num_hiden_layer_units=128
        self.num_output_units=2
        self.learning_rate=5e-5
        self.batchsize=20
        self.echo=150

        self.restore_model_path=restore_model_path
        self.log_dir=log_dir

        self.y_pred,self.train_step,self.accuracy=self.build_model()

        # self.doc = doc
        self.word_embedding=self.load_embedding(path=w2v_path)

        # self.init_state=self.init_state()
        # self.tree=self.tree_builder()

    def load_embedding(self,path):
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        return word_vectors

    #train model code
    def build_model(self):
        '''
        build a neural network.
        '''
        input_units=self.num_input_units
        hiden_layer=self.num_hiden_layer
        hiden_layer_units=self.num_hiden_layer_units
        output_units=self.num_output_units
        learning_rate=self.learning_rate

        self.sess = tf.InteractiveSession()
        with tf.name_scope("TreeBuilderNeuralNetwork-weights_and_biases"):
            W1 = tf.Variable(tf.truncated_normal([input_units, hiden_layer_units], stddev=0.1),name='W1')
            b1 = tf.Variable(tf.zeros([hiden_layer_units]),name='b1')
            W2 = tf.Variable(tf.truncated_normal([hiden_layer_units, output_units], stddev=0.1),name='W2')
            b2 = tf.Variable(tf.zeros([output_units]),name='b2')
        with tf.name_scope("TreeBuilderNeuralNetwork-placeholders"):
            self.keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
            self.x = tf.placeholder(tf.float32, [None, input_units],name='x')
            self.y_true = tf.placeholder(tf.float32, [None, output_units],name='label_true')
        with tf.name_scope("TreeBuilderNeuralNetwork-calculate_steps"):
            hidden1 = tf.nn.relu(tf.matmul(self.x, W1) + b1,name='hidden_layer')
            hidden1_drop = tf.nn.dropout(hidden1, self.keep_prob,name='hidden_layer_dropout')
            y_pred = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2,name='label_predition')

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_true * tf.log(y_pred), reduction_indices=[1]),name='cross_entropy')
            # L2正则化
            regularizer = tf.contrib.layers.l2_regularizer(1e-2)  # 损失函数
            regularization = regularizer(W1) + regularizer(W2)
            loss = cross_entropy + regularization

            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) #cross_entropy
            # tf.global_variables_initializer().run()
            correct_prediction = tf.equal(tf.argmax(self.y_true, 1), tf.argmax(y_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        return y_pred,train_step,accuracy

    def __shuffle_batch(self,data,label,batch_size):
        for i in range(len(label)):
            if label[i]==0:
                label[i]=[0,1]
            else:
                label[i] = [1, 0]
        batch_num=int(len(data)/batch_size)
        x_batch=[]
        y_batch=[]
        for i in range(batch_num):
            x_batch.append(data[i*batch_size:(i+1)*batch_size])
            y_batch.append(label[i * batch_size:(i + 1) * batch_size])
        return x_batch,y_batch

    def metrics(self,y_true, y_pred):
        metrics_score = {}
        ## y_true 格式 [[0,1],[0,1],[1,0]]
        ## y_pred 格式 [[0.1,0.9],[0.3,0.7],[0.8,0.2]]
        y_true_bin = [i[0] for i in list(y_true)]
        y_pred_bin = list((np.array([i[0] for i in list(y_pred)]) > 0.5) + 0)

        metrics_score['accuracy_score'] = skmetrics.accuracy_score(y_true_bin, y_pred_bin)  # 准确率
        metrics_score['average_precision_score'] = skmetrics.average_precision_score(y_true_bin, y_pred_bin)  # 平均准确率
        metrics_score['precision_score'] = skmetrics.precision_score(y_true_bin, y_pred_bin)  # 精确率(查准率)
        metrics_score['recall_score'] = skmetrics.recall_score(y_true_bin, y_pred_bin)  # 召回率(查全率)
        metrics_score['f1_score'] = skmetrics.f1_score(y_true_bin, y_pred_bin)

        # eg. true_label=[[0,1],[0,1],[1,0]]   predict_label=[[0.1,0.9],[0.3,0.7],[0.8,0.2]]  #完整的每个类别的得分矩阵
        # skm.roc_auc_score(np.array(true_label),np.array(predict_label))   >>>1.0
        metrics_score['roc_auc_score'] = skmetrics.roc_auc_score(np.array(y_true), np.array(y_pred))

        return metrics_score

    def train_model(self,train_path,val_path,test_path):

        #load dataset
        with open(train_path,'r') as f:
            train_data=f.read().strip().decode('utf-8').split('\n')
            train_x_y=[]
            for d in train_data:
                tmp=d.strip().split('##')
                w2v_features=tmp[0].strip().split()[0:80]
                human_features=[float(i) for i in tmp[0].strip().split()[80:]]
                train_x_y.append([[w2v_features,human_features],int(tmp[1].strip())])
        with open(val_path, 'r') as f:
            val_data = f.read().strip().decode('utf-8').split('\n')
            val_x_y = []
            for d in val_data:
                tmp = d.strip().split('##')
                w2v_features=tmp[0].strip().split()[0:80]
                human_features=[float(i) for i in tmp[0].strip().split()[80:]]
                val_x_y.append([[w2v_features,human_features],int(tmp[1].strip())])
        with open(test_path,'r') as f:
            test_data=f.read().strip().decode('utf-8').split('\n')
            test_x_y=[]
            for d in test_data:
                tmp=d.strip().split('##')
                w2v_features=tmp[0].strip().split()[0:80]
                human_features=[float(i) for i in tmp[0].strip().split()[80:]]
                test_x_y.append([[w2v_features,human_features],int(tmp[1].strip())])
        random.shuffle(train_x_y)
        random.shuffle(val_x_y)
        random.shuffle(test_x_y)

        saver = tf.train.Saver(max_to_keep=150)
        sess=self.sess
        train_step=self.train_step
        accuracy=self.accuracy
        word_embedding=self.word_embedding

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())

        print ('begin train')

        train_x_batch,train_y_batch=self.__shuffle_batch(data=[i[0] for i in train_x_y], label=[i[1] for i in train_x_y],
                                                       batch_size=self.batchsize)

        val_label = [] ##val
        for ii in val_x_y:
            if ii[1] == 0:
                val_label.append([0, 1])
            else:
                val_label.append([1, 0])
        feed_dict_val = {self.x: np.array([list(word_embedding[i[0]].flatten())+i[1] for i in [ii[0] for ii in val_x_y]], dtype=np.float32).reshape([len(val_x_y), -1]),#np.array([word_embedding[i[0]].flatten()+i[1] for i in [ii[0] for ii in val_x_y]], dtype=np.float32).reshape([len(val_x_y), -1]),
                     self.y_true: np.array(val_label, dtype=np.float32), self.keep_prob: 1}

        test_label = []  ##test
        for ii in test_x_y:
            if ii[1] == 0:
                test_label.append([0, 1])
            else:
                test_label.append([1, 0])
        feed_dict_test = {self.x: np.array([list(word_embedding[i[0]].flatten())+i[1] for i in [ii[0] for ii in test_x_y]], dtype=np.float32).reshape([len(test_x_y), -1]),#np.array([word_embedding[i] for i in [ii[0] for ii in test_x_y]], dtype=np.float32).reshape([len(test_x_y), -1]),
                     self.y_true: np.array(test_label, dtype=np.float32), self.keep_prob: 1}

        ## begin train
        for e in range(self.echo):
            train_acc=0
            for one_batch_x,one_batch_y in zip(train_x_batch,train_y_batch):
                feed_dict={self.x:np.array([list(word_embedding[i[0]].flatten())+i[1] for i in one_batch_x],dtype=np.float32).reshape([self.batchsize,-1]), #np.array([word_embedding[i] for i in one_batch_x],dtype=np.float32).reshape([self.batchsize,-1]),
                           self.y_true:np.array(one_batch_y,dtype=np.float32),self.keep_prob:0.80}
                tmp_acc,_=sess.run([accuracy,train_step], feed_dict=feed_dict)
                train_acc+=tmp_acc
            train_acc=train_acc/float(len(train_y_batch))

            val_y_true,val_y_pred = sess.run([self.y_true,self.y_pred], feed_dict=feed_dict_val)
            print (e,'train_acc=',train_acc,'\nval metrics:',self.metrics(val_y_true,val_y_pred))

            ##save model
            saver.save(sess, self.restore_model_path+"/model.ckpt", global_step=e)

        test_y_true, test_y_pred = sess.run([self.y_true, self.y_pred], feed_dict=feed_dict_test)
        print('\ntest metrics:', self.metrics(test_y_true, test_y_pred),'\n')
        print('training finish')
        writer.close()

    ##use model code
    def restore_model(self):
        path_model = self.restore_model_path
        saver = tf.train.Saver()
        # sess = self.sess
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(path_model))

    def _oracle(self,Stack_now,ParagraphList_now,Relation_previous):
        '''
        operate shift or reduce by using self._model
        :return:
        '''
        if Stack_now==['<root>'] and len(ParagraphList_now)>0:
            return 'shift'
        elif Stack_now==['<root>'] and len(ParagraphList_now)==0:
            return 'Done'
        elif Stack_now!=['<root>'] and len(ParagraphList_now)==0:
            return 'reduce'
        else:
            ## build  feed_dict
            # 数据使用
            #     Stack里面栈顶的3段，每段首尾5个词（不足补0）  3×（5+5）*50=1500
            #     Paragraph List里面的后3段，每段首尾5个词（不足补0） 3×（5+5）*50=1500
            #     Relation里面当前的 前一个head-->dependent 这2段的每段的首尾5个词（不足补0）   2×（5+5）*50=1000

            embedding_vocab_dict = dict(zip(self.word_embedding.vocab.keys(), self.word_embedding.vocab.keys()))

            stack_d = Stack_now[-3:]
            parag_list_d = ParagraphList_now[-3:]

            s = ['<null>'] * (3 - len(stack_d)) + stack_d  # 补齐后Stack
            p = parag_list_d + ['<null>'] * (3 - len(parag_list_d))  ##补齐后 Paragraph List

            if Relation_previous==[]:
                r = ['<null>', '<null>']
            else:
                r = [Relation_previous[0], Relation_previous[-1]]  ##Relation

            ##s p r 每个取首尾5个词
            l = s + p + r
            for e in range(len(l)):
                if type(l[e])==type({}): l[e]=l[e]['text']

            ##判断l是否遇到<root> <null>
            data_d = []
            for a in l:
                if a != '<root>' and a != '<null>':
                    sentence_list=[jj for jj in ' '.join(jieba.cut(a, cut_all=False)).split() if jj in embedding_vocab_dict]
                    if len(sentence_list) >= 5:
                        data_d.append(sentence_list[:5] + sentence_list[-5:])
                    else:
                        data_d.append(sentence_list + ['</s>'] * (5 - len(sentence_list)) * 2 + sentence_list)  ##不足补零
                elif a == '<root>':
                    data_d.append(['<root>'] * 10)
                elif a == '<null>':
                    data_d.append(['</s>'] * 10)  ##前五个，后五个

            input_x=self.word_embedding[list(chain.from_iterable(data_d))]
            hf=self._human_feature(Stack_now[-1],ParagraphList_now[-1],Relation_previous)
            input_x=list(input_x.reshape([-1]))+hf
            input_x=np.array(input_x,dtype=np.float32).reshape([1,-1])
            feed_dict={self.x:input_x,self.keep_prob:1.0}
            sess=self.sess
            y_pred=sess.run(self.y_pred, feed_dict=feed_dict)
            y_pred=y_pred.reshape([-1])
            # if -0.2<y_pred[0]-y_pred[1]<0.2:
            #     print('debug')
            #     #debug

            if y_pred[0]<y_pred[1]:  #[0,1] shift , [1,0]  reduce
                return 'shift'
            else:
                return 'reduce'

    def _human_feature(self,stack_last,para_list_last,previous_relation):
        '''
            is_bold: True/False  -->  1/0                                  1维
            pattern: 34(pattern.txt)+1(other)  -->  onehot [0,1,0,...,0]   35维
            font_size: 22.08~88.08  -->  除以100, 0~1                       1维
            tag:paragraph/table  -->   0/1                                 1维
            jc:left/center -->   0/1                                       1维
            indent: 0~64   -->   onehot [0,1,0,...,0]                      65维
            number: 0~39  -->  除以40 , 0~1                                 1维
                                                                        合计 =105维  使用4行 105*4=420维
        '''
        pattrens='''pattern:第[零〇一二三四五六七八九十百]+部分
                pattern:（[1-9][0-9]?）
                pattern:第[零〇一二三四五六七八九十百]+条
                pattern:[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?
                pattern:([1-9][0-9]?)
                pattern:[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?
                pattern:[a-z]
                pattern:[1-9][0-9]?．
                pattern:[1-9][0-9]?）
                pattern:附件[零〇一二三四五六七八九十百]+：
                pattern:(XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)+
                pattern:（[零〇一二三四五六七八九十百]+）
                pattern:[1-9][0-9]?.[1-9][0-9]?
                pattern:([零〇一二三四五六七八九十百]+)
                pattern:[零〇一二三四五六七八九十百]+、
                pattern:[-]+
                pattern:(xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)+．
                pattern:[A-Z]
                pattern:[1-9][0-9]?
                pattern:[1-9][0-9]?.
                pattern:[①-⑳]+
                pattern:[1-9][0-9]?)
                pattern:(XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)+：
                pattern:第[1-9][0-9]?章
                pattern:[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?.[1-9][0-9]?
                pattern:[零〇一二三四五六七八九十百]+
                pattern:(xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)+
                pattern:[A-Z]：
                pattern:第[零〇一二三四五六七八九十百]+章
                pattern:[a-z]．
                pattern:
                pattern:注[1-9][0-9]?：
                pattern:附件[零〇一二三四五六七八九十百]+
                pattern:[1-9][0-9]?、'''.replace('pattern:','').replace(' ','').split('\n')

        if not para_list_last:
            para_list_last=['<null>']

        if previous_relation:
            l=[stack_last]+[para_list_last]+[previous_relation[0]]+[previous_relation[-1]]
        else:
            l = [stack_last] + [para_list_last]+['<null>', '<null>']

        whole_hf=[]
        for i in l:
            if i=='<null>' or i=='<root>':
                # hf=['<NoneHF>']*12
                hf = [0] * 105
            elif i['other_feature']['tag']=='table':
                # hf=['tag:table']+['<NoneHF>']*11
                hf = [0] * 105
                hf[47]=1
            else:
                hf=[]
                other_features=i['other_feature']
                if other_features['is_bold']==True:
                    hf.append(1)
                else:hf.append(0)

                if other_features['pattern'] in pattrens:
                    tmp=[0]*35
                    tmp[pattrens.index(other_features['pattern'])]=1
                    hf+=tmp
                else:
                    tmp=[0]*35
                    tmp[-1]=1
                    hf+=tmp

                hf.append(other_features['font_size']/100.0)

                hf.append(0)  ## tag : paragraph

                if other_features['jc']=='left':
                    hf.append(0)
                else:
                    hf.append(1)

                indent=int(other_features['indent'])
                tmp = [0] * 65
                if indent<65:
                    tmp[indent]=1
                hf+=tmp

                hf.append(other_features['number']/40.0)

            whole_hf += hf

        return whole_hf

    def _build_states(self,doc):
        '''
        做成  Stack ParagraphList   Action       Relation 格式
            <root>  P0~PN        shift/reduce    Pn --> Pm/[]

        other_features具体如下：
            is_bold: True/False  -->  1/0                                  1维
            pattern: 34(pattern.txt)+1(other)  -->  onehot [0,1,0,...,0]   35维
            font_size: 22.08~88.08  -->  除以100, 0~1                       1维
            tag:paragraph/table  -->   0/1                                 1维
            jc:left/center -->   0/1                                       1维
            indent: 0~64   -->   onehot [0,1,0,...,0]                      65维
            number: 0~39  -->  除以40 , 0~1                                 1维
                                                                        合计 =105维  使用4行 105*4=420维
        '''
        def other_feature(doc_to_jsoni):
            if doc_to_jsoni['tag']=='paragraph':
                return {'is_bold': doc_to_jsoni['style']['is_bold'],
                 'pattern': doc_to_jsoni['number_list']['pattern'],
                 'font_size': doc_to_jsoni['style']['font_size'],
                 'tag': doc_to_jsoni['tag'],
                 'jc': doc_to_jsoni['style']['jc'],
                 'indent': doc_to_jsoni['style']['indent'],
                 'number': doc_to_jsoni['number_list']['number']
                 }
            elif doc_to_jsoni['tag']=='table':
                return {'tag':'table','text':str(doc_to_jsoni['cells']),'cells':doc_to_jsoni['cells']}

        #todo 接DocumentCorrection类do_correction()
        # #doc_contents = self.do_correction([str(i.to_string(number_list=True, indent=False).encode('utf-8')) for i in doc.contents])
        doc_to_json=doc.to_json()
        doc_contents=[str(i.to_string(number_list=True, indent=False).encode('utf-8')) for i in doc.contents]
        doc_contents=[{'text':doc_contents[i],'ind':i,'other_feature':other_feature(doc_to_json['contents'][i])}
                      for i in range(len(doc_contents))]
        for i in doc_contents:
            if i['other_feature']['tag']=='table':
                i['other_feature']['table_text']=copy.deepcopy(i['text'])
                i['text']=i['other_feature']['text']  ##引用

        Stack = ['<root>']
        ParagraphList = copy.deepcopy(doc_contents)
        Relation = []
        Action = 'shift'

        states=[]

        while True:
            if Stack==['<root>'] and len(ParagraphList)!=0:
                Action = 'shift'   ##Action 强制设为shift
            else:##__oracle 根据所有的configuration 即states判断 shift还是reduce
                Action=self._oracle(Stack_now=Stack,ParagraphList_now=ParagraphList,Relation_previous=Relation)

            ##根据Action来执行当前行操作
            if Action == 'shift':
                Relation = []
                state = [Stack, ParagraphList, Action, Relation]
                states.append(copy.deepcopy(state))

                Stack.append(ParagraphList.pop())   ##为下一个准备

            elif Action == 'reduce':
                if len(Stack) > 2:
                    Relation = [Stack[-1],'-->',Stack[-2]]
                else:
                    Relation = [Stack[-2], '-->', Stack[-1]]

                state = [Stack, ParagraphList, Action, Relation]
                states.append(copy.deepcopy(state))

                if len(Stack) > 2:
                    Stack.pop(-2)
                else:
                    Stack.pop(-1)
            elif Action == 'Done':
                Relation = []
                state = [Stack, ParagraphList, Action, Relation]
                states.append(copy.deepcopy(state))
                break  ##退出
        return states

    def build_tree(self,doc,other_features=False):
        '''
        :param doc: doc=Document.from_file
        :param other_features: if True,return human_features; else not return
        :return: json to string type tree
        '''

        states=self._build_states(doc=doc)
        # doc_contents = [i.to_string(number_list=True, indent=False) for i in doc.contents]
        # doc_contents=[{'text':doc_contents[i],'ind':i} for i in range(len(doc_contents))]

        relations=[i[-1] for i in states if i[-1]]
        relations_tmp=[]
        for i in range(len(relations)-1):
            if type(relations[i][0])==type({}):a=relations[i][0]['text']
            else:a=relations[i][0]
            if type(relations[i+1][0])==type({}):b=relations[i+1][0]['text']
            else:b=relations[i+1][0]

            if a==b:
                relations_tmp.append(relations[i])
            else:
                relations_tmp.append(relations[i])
                relations_tmp.append([])
        relations_tmp.append(relations[-1])

        ##resort relations_tmp
        tmp=[]
        relations_tmp=relations_tmp+[[]]

        while relations_tmp:
            nullindex=relations_tmp.index([])
            tmp.append(relations_tmp[:nullindex])
            relations_tmp=copy.deepcopy(relations_tmp[nullindex + 1:])
        tmp=list(reversed(tmp))
        relations=list(chain.from_iterable(tmp))

        jsonlist=[]
        dict_ind_depth={}
        dict_ind_depth[-1]=0
        for i in relations:
            if type(i[0])==type({}):a=i[0]['ind']
            else:a=-1
            if type(i[-1])==type({}):b=i[-1]['ind']
            else:b=-1
            father_depth=dict_ind_depth[a]
            child=i[-1]
            child['depth']=father_depth+1
            jsonlist.append(child)
            dict_ind_depth[b]=father_depth+1

        jsonlist=sorted(jsonlist, key=lambda x: x['ind'])

        for i in jsonlist:
            if i['other_feature']['tag']=='table':
                i['text']=i['other_feature']['table_text']
            if other_features == False:
                i.pop('other_feature')

        return json.dumps(jsonlist)

    ### functions for json

    def __split_documents_for_json(self,js):
        annotations = js[u'annotations']
        documents = js[u'documents']
        r = []
        for i in range(len(annotations)):
            document_id = annotations[i][u'document_id']
            content_id = annotations[i][u'content_id']
            try:
                haven = documents[document_id][u'contents'][content_id].get(u'annotation')
                if haven:
                    haven.append(annotations[i])
                    documents[document_id][u'contents'][content_id][u'annotation'] = haven
                else:
                    documents[document_id][u'contents'][content_id][u'annotation'] = [annotations[i]]
            except:
                print(document_id, content_id, ' skipped')
        for d in documents:
            documenti = d[u'contents']
            r.append(documenti)
        return r

    def _build_one_states_for_json(self,doc):
        def other_feature(doci):
            if doci['tag']=='paragraph':
                return {'is_bold': doci['style']['is_bold'],
                        'pattern': doci['number_list']['pattern'],
                        'font_size': doci['style']['font_size'],
                        'tag': doci['tag'],
                        'jc': doci['style']['jc'],
                        'indent': doci['style']['indent'],
                        'number': doci['number_list']['number'],
                        'annotation':doci.get('annotation')}
            elif doci['tag']=='table':
                return {'tag':'table','text':json.dumps((doci['cells'])),'annotation':doci.get('annotation')}

        for i in range(len(doc)) :
            if doc[i]['tag'].lower()=='table':
                doc[i]['text']={}
                doc[i]['text']['text']=json.dumps(doc[i]['cells'])
                doc[i]['number_list'] = {}
                doc[i]['number_list']['text']=''

        doc_contents=[{'text_no_number':doc[i]['text']['text'],
                       'text':(doc[i]['number_list']['text']+' '+doc[i]['text']['text']).replace('  ',' ').strip(),
                       'ind':i,'other_feature':other_feature(doc[i])}
                      for i in range(len(doc))]

        Stack = ['<root>']
        ParagraphList = copy.deepcopy(doc_contents)
        Relation = []
        Action = 'shift'

        states=[]

        while True:
            if Stack==['<root>'] and len(ParagraphList)!=0:
                Action = 'shift'   ##Action 强制设为shift
            else:##__oracle 根据所有的configuration 即states判断 shift还是reduce
                Action=self._oracle(Stack_now=Stack,ParagraphList_now=ParagraphList,Relation_previous=Relation)

            ##根据Action来执行当前行操作
            if Action == 'shift':
                Relation = []
                state = [Stack, ParagraphList, Action, Relation]
                states.append(copy.deepcopy(state))

                Stack.append(ParagraphList.pop())   ##为下一个准备

            elif Action == 'reduce':
                if len(Stack) > 2:
                    Relation = [Stack[-1],'-->',Stack[-2]]
                else:
                    Relation = [Stack[-2], '-->', Stack[-1]]

                state = [Stack, ParagraphList, Action, Relation]
                states.append(copy.deepcopy(state))

                if len(Stack) > 2:
                    Stack.pop(-2)
                else:
                    Stack.pop(-1)
            elif Action == 'Done':
                Relation = []
                state = [Stack, ParagraphList, Action, Relation]
                states.append(copy.deepcopy(state))
                break  ##退出
        return states

    def _resort_state(self,states):

        relations = [i[-1] for i in states if i[-1]]
        relations_tmp = []
        for i in range(len(relations) - 1):
            if type(relations[i][0]) == type({}):
                a = relations[i][0]['text']
            else:
                a = relations[i][0]
            if type(relations[i + 1][0]) == type({}):
                b = relations[i + 1][0]['text']
            else:
                b = relations[i + 1][0]

            if a == b:
                relations_tmp.append(relations[i])
            else:
                relations_tmp.append(relations[i])
                relations_tmp.append([])
        relations_tmp.append(relations[-1])

        ##resort relations_tmp
        tmp = []
        relations_tmp = relations_tmp + [[]]

        while relations_tmp:
            nullindex = relations_tmp.index([])
            tmp.append(relations_tmp[:nullindex])
            relations_tmp = copy.deepcopy(relations_tmp[nullindex + 1:])
        tmp = list(reversed(tmp))
        relations = list(chain.from_iterable(tmp))
        # relations 重排序，先按照father‘s ind，再按child’s ind
        # relations=[relations[0]]+sorted(relations[1:], key=lambda x: (x[0]['ind'],x[2]['ind']))  ###todo 加上后有报错
        relations = sorted(relations, key=lambda x: len(str(x[0])))
        for i in range(len(relations)):
            if relations[i][0]!='<root>':
                break
        if i==0:
            relations = sorted(relations, key=lambda x: (x[0]['ind'],x[2]['ind']))
        elif i==1:
            relations =[relations[0]] + sorted(relations[1:], key=lambda x: (x[0]['ind'], x[2]['ind']))
        else:
            relations = sorted(relations[:i], key=lambda x: x[2]['ind']) + sorted(relations[i:], key=lambda x: (x[0]['ind'], x[2]['ind']))

        jsonlist = []
        dict_ind_depth = {}
        dict_ind_depth[-1] = 0
        for i in relations:
            if type(i[0]) == type({}):
                a = i[0]['ind']
            else:
                a = -1
            if type(i[-1]) == type({}):
                b = i[-1]['ind']
            else:
                b = -1
            father_depth = dict_ind_depth.get(a)
            if  father_depth==None:
                father_depth=0
            child = i[-1]
            child['depth'] = father_depth + 1
            jsonlist.append(child)
            dict_ind_depth[b] = father_depth + 1

        jsonlist = sorted(jsonlist, key=lambda x: x['ind'])

        return json.dumps(jsonlist)

    def __build_states_from_json(self,js):

        docs=self.__split_documents_for_json(js)
        states=[]
        for doc in docs:  ##todo delete[]
            try:
                states.append(self._resort_state(self._build_one_states_for_json(doc)))
                print('\rBuild tree for json: %d%%'%((1+docs.index(doc))*100.0/len(docs)),end='')
            except:
                print('Error in json files\' %d ,skiped it',docs.index(doc))
                continue
        print('\n')
        return states

    def build_tree_from_json(self,json_path='Corpus_data/Corpus.json'):

        with open(json_path,'r') as f:
            js=json.loads(f.read())

        states = self.__build_states_from_json(js=js)
        return states

def main():
    '''
    usage:
    1. build training dataset steps:
        bt=BuildTrainDataset(json_path='/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/json_indent_numberlist/',
                            savedata_path='/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/json_indent_numberlist/train_data/',
                            wordembedding_dir='/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/zhwiki_finance_simple.sg_50d.word2vec')
        bt.bulid()

    2. train model steps:
        nn=TreeBuilderNeuralNetwork(restore_model_path='Tree_model_saved/',log_dir='./log/',w2v_path='zhwiki_finance_simple.sg_50d.word2vec')
        train_path='/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/json_indent_numberlist/train_data/train_dataset.txt'
        val_path='/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/json_indent_numberlist/train_data/val_dataset.txt'
        test_path='/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/json_indent_numberlist/train_data/test_dataset.txt'
        nn.train_model(train_path=train_path,val_path=val_path,test_path=test_path)


    3. us trained model steps:
        nn=TreeBuilderNeuralNetwork(restore_model_path='Tree_model_saved/',log_dir='./log/',w2v_path='zhwiki_finance_simple.sg_50d.word2vec')
        doc=Document.from_file('/home/garret/Desktop/GeneralParser/GeneralParser/Src/pdfs/e554b2b1-0332-4b55-97ca-069445772e66_terms.pdf')
        nn.restore_model()
        tree=nn.build_tree(doc=doc)

    '''

    ## use model
    nn=TreeBuilderNeuralNetwork(restore_model_path='Tree_model_saved/',log_dir='./log/',w2v_path='zhwiki_finance_simple.sg_50d.word2vec')
    nn.restore_model()

    doc=Document.from_file('/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/pdfs/72d505c2-f4dc-4a77-a9dc-dc9ec3680477_terms.pdf')
    tree=nn.build_tree(doc=doc)
    # tree=nn.build_tree_from_json()
    ##can build tree for more than one doc by for ... doc=... tree=nn.build_tree(doc=doc)

    with open('/home/garret/Desktop/nn-72d505c2-f4dc-4a77-a9dc-dc9ec3680477_terms.txt','w') as f:
        f.write(tree)



if __name__ == "__main__":
    main()