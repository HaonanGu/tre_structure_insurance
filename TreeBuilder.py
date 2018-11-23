# -*- coding: utf-8 -*-
from __future__ import print_function
import copy
import json
import cPickle
from math import log, exp
from Doc.Document import Document
from Parser.Scorer import Scorer
from Tools.FileIO import FileIO


__metaclass__ = type


class TreeBuilder:
    """
    Build tree from a structured document. Tree is built layer-wise. At each layer, unlabelled paragraphs
    are scored. The score is converted to the probability of labelling the paragraph for the current layer. The top
    k_running probabilities are kept during the run. When a layer is done, top k_layer results with highest
    probabilities are selected for the next run. This is a greedy algorithm.
    """
    def __init__(self, length, k_layer, k_running, scorer):
        """
        Ctor: build tree and store the best result (highest probability)
        :param length: int, number of paragraphs
        :param k_layer, k_running: int, int, see above
        :param scorer: an instance of Scorer class that implements two methods: reset and score. See the class
        definition for more info.
        """
        self._scorer = scorer
        self._length = length
        self._k_layer = k_layer
        self._k_running = k_running
        self._label_lists = self._build()
        self._depths, self._log_prob = self._label_lists[0]
        self._sections, self._idx_to_section = TreeBuilder._make_sections(self._depths)
        self._parents, self._children = TreeBuilder._mark_parents(self._depths)

    @staticmethod
    def save(instance, filename):
        with open(filename, "wb") as save_file:
            cPickle.dump(instance, save_file, 2)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as save_file:
            instance = cPickle.load(save_file)
        return instance

    @property
    def depths(self):
        """
        Return the depths of paragraphs
        :return: a list of int, corresponding to the depth of each paragraph in the tree
        """
        return self._depths

    @property
    def sections(self):
        return self._sections

    def section_of(self, idx):
        return self._idx_to_section[idx]

    def depth(self, idx):
        return self._depths[idx]

    def parent(self, idx):
        return self._parents[idx]

    def children(self, idx):
        return self._children[idx]

    def is_leaf(self, idx):
        return len(self._children[idx]) == 0

    def to_json(self, doc):
        root = {"text": "root", "children": [], "depth": -1}
        stack = [root]
        for content, depth in zip(doc.contents, self._depths):
            node = {"text": content.to_string(True, False), "children": [], "depth": depth}
            while stack[-1]["depth"] >= depth:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
        self._clean_up(root)
        return json.dumps(root).decode("raw_unicode_escape").encode("utf-8")

    def _clean_up(self, node):
        node.pop("depth")
        for child in node["children"]:
            self._clean_up(child)
        if len(node["children"]) == 0:
            node.pop("children")

    def _build(self):
        curr_depth = 1
        depths = [([None for _ in range(self._length)], 0.0)]
        finish = False
        while not finish:
            finish = True
            log_probs = []
            for depth, log_prob in depths:
                next_label_lists = self._build_layer(depth, curr_depth)
                if len(next_label_lists) > 1:
                    finish = False
                for next_label_list, next_log_prob in next_label_lists:
                    log_probs.append((depth, next_label_list, log_prob + next_log_prob))
            log_probs.sort(key=lambda x: x[2], reverse=True)
            next_layer = []
            for top_k_log_prob in log_probs[:self._k_layer]:
                next_layer.append(self._merge(curr_depth, *top_k_log_prob))
            depths = next_layer
            curr_depth += 1
        return depths

    def _merge(self, depth, curr_list, next_list, log_prob):
        new_list = copy.deepcopy(curr_list)
        for idx in next_list:
            new_list[idx] = depth
        return new_list, log_prob

    def _build_layer(self, depths, curr_depth):
        self._scorer.reset(depths, curr_depth)
        labels = [([], 0.0)]
        for idx, prev_label in enumerate(depths):
            next_labels = []
            if prev_label is not None:
                continue
            for curr_label, log_prob in labels:
                score = self._scorer.score(curr_label, curr_depth, idx)
                pos_prob = 1.0 / (1.0 + exp(-score))
                neg_prob = 1.0 - pos_prob + 1e-10
                log_prob_pos = log(pos_prob)
                log_prob_neg = log(neg_prob)
                next_labels.append((curr_label + [idx], log_prob + log_prob_pos))
                next_labels.append((curr_label + [], log_prob + log_prob_neg))
            next_labels.sort(key=lambda x: x[1], reverse=True)
            labels = next_labels if len(next_labels) <= self._k_running else next_labels[:self._k_running]
        return labels

    @staticmethod
    def _mark_parents(depths):
        stack = [(None, -1)]
        parents = []
        children = [[] for _ in range(len(depths))]
        for idx, depth in enumerate(depths):
            while stack[-1][1] >= depth:
                stack.pop()
            parents.append(stack[-1][0])
            if stack[-1][0] is not None:
                children[stack[-1][0]].append(idx)
            stack.append((idx, depth))
        return parents, children

    @staticmethod
    def _make_sections(depths):
        max_depth = max(depths)
        running_sections = [[] for _ in range(max_depth + 1)]
        sections = []
        for idx, depth in enumerate(depths):
            for child_section_depth in range(depth + 1, max_depth + 1):
                if len(running_sections[child_section_depth]) == 0:
                    continue
                sections.append(running_sections[child_section_depth])
                running_sections[child_section_depth] = []
            running_sections[depth].append(idx)
        for depth in range(max_depth + 1):
            if len(running_sections[depth]) > 0:
                sections.append(running_sections[depth])
        idx_to_section = [None for _ in range(len(depths))]
        for section in sections:
            for idx in section:
                idx_to_section[idx] = section
        return sections, idx_to_section


def main():
    pathname = "/home/jiqing/Annotation/pdfs/"
    output_pathname = "/home/jiqing/Annotation/Annotated/"
    filenames = FileIO.files_in_path(pathname, "pdf")

    for filename in filenames:
        try:
            doc = Document.from_file(pathname + filename)
            scorer = Scorer(doc.contents)
            tree_builder = TreeBuilder(len(doc.contents), 3, 20, scorer)
            with open(output_pathname + filename[:-4] + ".txt", "w") as save_file:
                output = []
                for idx, (depth, content) in enumerate(zip(tree_builder.depths, doc.contents)):
                    output_content = {"idx": idx, "depth": depth, "text": content.to_string(True, False)}
                    output.append(output_content)
                save_file.write(json.dumps(output).decode("raw_unicode_escape").encode("utf-8"))
                save_file.flush()
        except:
            pass


if __name__ == "__main__":
    main()

