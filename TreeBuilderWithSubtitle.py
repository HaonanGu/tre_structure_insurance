# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('../')

import tensorflow as tf
from Doc.Document import Document
from StanfordNLP.StanfordNLP import StanfordNLP
from TreeBuilderNeuralNetwork import TreeBuilderNeuralNetwork,DocumentCorrection
from RoughClassifier import RoughClassifier
import json
import copy
import numpy as np

class TreeBuilderWithSubtitle():
    _roughclassifier_dim = 31

    def __init__(self,path_treebuilder_model='Tree_model_saved/',
                path_roughclassifier_model='Classifier_model/RoughClassifier_model/',
                w2v_path='zhwiki_finance_simple.sg_50d.word2vec'):
        tf.reset_default_graph()
        self.roughclassifier=RoughClassifier(model_path=path_roughclassifier_model,w2v_path=w2v_path)
        self.roughclassifier.restore()

        tf.reset_default_graph()
        self.nntreebuilder = TreeBuilderNeuralNetwork(restore_model_path=path_treebuilder_model, log_dir='',w2v_path=w2v_path)
        self.nntreebuilder.restore_model()

    @staticmethod
    def _radiate_subtitle(y_names,y_preds,depths):
        if len(y_names)==len(y_preds)==len(depths):
            st_depth=0
            st_y_name = ''
            st_y_pred = 0
            for i in range(len(y_names)):
                if y_names[i]==u'title':
                    st_y_name=y_names[i]
                    st_y_pred=y_preds[i]
                    st_depth = depths[i]
                elif y_names[i] !=u'None' and st_y_name!=u'title' and depths[i]>=st_depth+1:
                    st_y_name=y_names[i]
                    st_y_pred=y_preds[i]
                    st_depth = depths[i]
                elif y_names[i] ==u'None' and st_y_name!=u'title' and depths[i]>=st_depth+1:
                    y_names[i]=st_y_name
                    y_preds[i]=st_y_pred
                else:
                    st_y_name = y_names[i]
                    st_y_pred = y_preds[i]
                    st_depth=depths[i]

        else:
            raise Exception('length error, make sure inputs have same length ( y_names,y_preds and depths)')

        return y_names,y_preds

    @staticmethod
    def _neighbourhood_judgment(y_names,y_preds):
        if len(y_names) == len(y_preds):
            for i in range(len(y_names)):
                if y_names[i]==u'title' :
                    continue
                if i==0: ##start
                    three_neighbourhoods = y_names[0:i + 2]
                elif i==len(y_names)-1: ##end
                    three_neighbourhoods = y_names[i - 1:i]
                else:
                    three_neighbourhoods = y_names[i-1:i+2]

                if three_neighbourhoods.count(y_names[i])<2:
                    y_names[i]=u'None'
                    y_preds[i]=[0.0]*TreeBuilderWithSubtitle._roughclassifier_dim
                if y_names[i]==u'None':  ##for [... A None A...] --> [... A A A ...]
                    if i == 0 or i ==len(y_names)-1:
                        continue
                    elif y_names[i-1]==y_names[i+1]!=u'None':
                        y_names[i]=y_names[i-1]
                        y_preds[i]=y_preds[i-1]
        else:
            raise Exception('length error, make sure inputs have same length ( y_names and y_preds) ')
        return y_names,y_preds

    def build_tree_with_subtitle(self,file_path='train_dataset/pdfs/72d505c2-f4dc-4a77-a9dc-dc9ec3680477_terms.pdf'):
        doc = Document.from_file(file_path)
        tree = self.nntreebuilder.build_tree(doc=doc,other_features=True)
        tree=json.loads(tree)

        doc_contents=[ii['text'] for ii in tree]
        doc_contents_merge_title,tag_has_title=DocumentCorrection.merge_titles(doc_contents=doc_contents,span=-1)

        doc_contents=[str(i.to_string(number_list=False, indent=False).encode('utf-8')) for i in doc.contents]
        for i in range(len(tree)):   ##delete tree paragarph's number
            text1=tree[i]['text']
            text2=doc_contents[i]
            tree[i]['text_with_number']=text1
            if type(text2)!=type(u''):text2=text2.decode('utf-8')
            if text1!=text2:
                tree[i]['text']=text2

        if not tag_has_title: ##无标题
            tree_content=tree
            split_ind=0
        else:##有标题
            split_ind=1+doc_contents_merge_title[0].count('<title_enter>')
            tree_content = tree[split_ind:]

        tree, sub_titles = self._build_tree_with_subtitle_main(tree, tree_content, split_ind)

        return tree,sub_titles

    def _build_tree_with_subtitle_main(self,tree,tree_content,split_ind):
        sub_titles=[u'合同构成',u'合同成立与生效',u'投保范围',u'保险责任',u'保险金额',u'保险费',u'责任免除',u'犹豫期',u'保险期间',u'受益人',
                    u'保险事故通知',u'保险金申请',u'等待期',u'无理赔奖励',u'责任延续',u'附加规则',u'合同解除',u'合同解除权的限制',u'诉讼时效',
                    u'宽限期',u'退保费用',u'部分领取手续费',u'最低保证利率',u'保险费率的调整',u'续保',u'初始费用',u'保单管理费',u'保单红利',
                    u'保单贷款',u'名词解释',u'None']
        y_names=[u'title']*split_ind
        y_preds=[[0.0]*TreeBuilderWithSubtitle._roughclassifier_dim]*split_ind

        for t in tree_content:
            if t['other_feature']['tag']=='table':
                y_names.append(u'None')
                y_preds.append([0.0]*TreeBuilderWithSubtitle._roughclassifier_dim)
                continue
            segments=StanfordNLP.segment(t['text'])
            text = RoughClassifier.segment_sentence(t['text'], copy.deepcopy(segments))
            pattern=t['other_feature']['pattern']
            try:
                y_pred=self.roughclassifier.usage(text=text,segments=segments,pattern=pattern)
            except:
                y_pred=[[0.0]*self._roughclassifier_dim]
                y_pred[0][-1]=1.0
                print('整句词都未在字典里发现：',text,'   请添加对应词向量')

            y_preds.append(list(y_pred[0]))
            y_names.append(sub_titles[np.argmax(y_pred)])

        depths=[i['depth'] for i in tree]
        y_names,y_preds=TreeBuilderWithSubtitle._radiate_subtitle(y_names, y_preds, depths)
        y_names, y_preds = TreeBuilderWithSubtitle._neighbourhood_judgment(y_names, y_preds)

        for i in range(len(tree)):
            tree[i]['subtitle'] = {}
            tree[i]['subtitle']['predit_feature']=y_preds[i]
            tree[i]['subtitle']['predit_name']=y_names[i]

        return tree,sub_titles

    def build_tree_with_subtitle_from_json(self,json_path='Corpus_data/Corpus.json'):

        trees = self.nntreebuilder.build_tree_from_json(json_path)

        rt_tree=[]
        for tree in trees:
            tree=json.loads(tree)

            doc_contents=[ii['text'] for ii in tree]
            doc_contents_merge_title,tag_has_title=DocumentCorrection.merge_titles(doc_contents=doc_contents,span=-1)

            doc_contents=[ii['text_no_number'] for ii in tree]
            for i in range(len(tree)):   ##delete tree paragarph's number
                text1=tree[i]['text']
                text2=doc_contents[i]
                tree[i]['text_with_number']=text1
                if type(text2)!=type(u''):text2=text2.decode('utf-8')
                if text1!=text2:
                    tree[i]['text']=text2

            if not tag_has_title: ##无标题
                tree_content=tree
                split_ind=0
            else:##有标题
                split_ind=1+doc_contents_merge_title[0].count('<title_enter>')
                tree_content = tree[split_ind:]

            tree, sub_titles=self._build_tree_with_subtitle_main(tree,tree_content,split_ind)
            rt_tree.append(tree)

        return rt_tree,sub_titles

    @staticmethod
    def example_json():
        TS = TreeBuilderWithSubtitle()
        trees, subtitle_names = TS.build_tree_with_subtitle_from_json(json_path='Corpus_data/Corpus.json')
        return trees,subtitle_names

    @staticmethod
    def example():
        T=TreeBuilderWithSubtitle()
        return T.build_tree_with_subtitle()

if __name__ == '__main__':

    trees, subtitle_names=TreeBuilderWithSubtitle.example_json()
    tree,subtitle_names=TreeBuilderWithSubtitle.example()

    print ('debug')  ## [i['text_with_number'] for i in tree]  ## [i['subtitle']['predit_name'] for i in tree]