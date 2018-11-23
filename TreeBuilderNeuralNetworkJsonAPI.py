# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append('../')

import copy
import json
import tensorflow as tf
from Doc.Document import Document
from TreeBuilderNeuralNetwork import TreeBuilderNeuralNetwork

__metaclass__ = type

class TreeBuilderNeuralNetworkJsonAPI(TreeBuilderNeuralNetwork):
    def __init__(self,restore_model_path='Tree_model_saved/',log_dir='./log/',w2v_path='zhwiki_finance_simple.sg_50d.word2vec'):
        TreeBuilderNeuralNetwork.__init__(self,restore_model_path=restore_model_path,log_dir=log_dir,w2v_path=w2v_path)

    def json_api(self,json_obj):
        '''
        需要先构造模型和restore模型
        :param json_obj: 与doc.to_json()相同的格式
        :return: 增加depth后的json_obj
        '''
        doc = json_obj['contents']
        tree=self._resort_state(states=self._build_one_states_for_json(doc=doc))
        tree=json.loads(tree)
        rt_json_obj=copy.deepcopy(json_obj)
        for i in range(len(tree)):
            rt_json_obj['contents'][i]['depth']=tree[i]['depth']
        return rt_json_obj

    @staticmethod
    def run_json_api(json_obj):
        '''
        每次使用都导入一次模型，速度慢，用于少数运算
        :return:
        '''
        tf.reset_default_graph()
        nnjson = TreeBuilderNeuralNetworkJsonAPI(restore_model_path='Tree_model_saved/', log_dir='./log/',
                                                 w2v_path='zhwiki_finance_simple.sg_50d.word2vec')
        nnjson.restore_model()
        rt_jsn = nnjson.json_api(json_obj=json_obj)
        del nnjson
        return rt_jsn

def main():
    ## usage methods:
    ## 1 You must have a json type input, type likes doc.to_json()
    doc=Document.from_file('/home/garret/Desktop/GeneralParser/GeneralParser/Src/Parser/train_dataset/pdfs/72d505c2-f4dc-4a77-a9dc-dc9ec3680477_terms.pdf')

    ## 2 You can restore model, once be enough.
    # tf.reset_default_graph()
    # nnjson=TreeBuilderNeuralNetworkJsonAPI(restore_model_path='Tree_model_saved/',log_dir='./log/',w2v_path='zhwiki_finance_simple.sg_50d.word2vec')
    # nnjson.restore_model()

    ## 3 Build tree with depth after restoring model. You can reuse the model and not need to restore it again.
    # jsn=nnjson.json_api(json_obj=doc.to_json())

    ## 4 Or, you can use run_json_api() after step 1 when the workload is not too great. Don't use it when step 2,3 are runing.
    jsn=TreeBuilderNeuralNetworkJsonAPI.run_json_api(json_obj=doc.to_json())

    print('done')


if __name__ == "__main__":
    main()