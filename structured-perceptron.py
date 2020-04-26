# author: Marjan Kamyab
# email: marjankamyab@ymail.com

from ingest import load_conll,Corpus
from hw2_utils import SequenceModel, extract_sentence_features
from random import shuffle, seed
from collections import Counter, defaultdict
from operator import itemgetter
import numpy as np

seed(0)

class Structured_Perceptron(SequenceModel):
    def __init__(self, epoch:int=1):
        self.epoch = epoch
        self.labels = []
        self.feature_set = None
        self.initial = defaultdict(int)
        self.emission = defaultdict(lambda: defaultdict(int))
        self.transition = defaultdict(lambda: defaultdict(int))
        self.start ='<s>'


    def set_parameters(self, train_data:Corpus) -> None:
        labels = []
        feature_set = []
        for i,doc in enumerate(train_data):
            sentences = doc.sentences
            tags = doc.labels
            for j,sent in enumerate(sentences):
                tag_sequence = tags[j]
                features = extract_sentence_features(sent)
                for k,token in enumerate(sent):
                    labels.append(tag_sequence[k])
                    feature_set += list(features[k].keys())
        labels = dict(Counter(labels))
        self.labels = [key for key,label in reversed(sorted(labels.items(), key=itemgetter(1)))]
        self.feature_set= set(feature_set)
        #weight initialization
        for label in self.labels:
            #initial transition weights
            self.initial[label] = 0
            #transition weights
            for next_label in self.labels:
                self.transition[label][next_label] = 0
            #emission weights
            for feat in self.feature_set:
                self.emission[label][feat] = 0


    def train(self, train_data: Corpus) -> None:
        gold_labels=[]
        pred_labels = []
        train_data= list(train_data)
        self.set_parameters(train_data)
        #scale = 0
        for i in range(self.epoch):
            shuffle(train_data)
            print("epoch: " + str(i+1))
            for doc in train_data:
                sentences = doc.sentences
                labels = doc.labels
                for j,sent in enumerate(sentences):
                    #scale += 1
                    feature_seq = extract_sentence_features(sent)
                    gold = labels[j]
                    predicted = self.viterbi(feature_seq)
                    #print(predicted)
                    #print(gold)
                    #print()
                    gold_labels.extend(gold)
                    pred_labels.extend(predicted)
                    for k,tok in enumerate(sent):
                        if gold[k] != predicted[k]:
                            self.update(predicted, gold, feature_seq, k)

        # averaging the parameters
        """for label in self.labels:
            self.initial[label] /= scale
            self.emission[label] = {k: (v/scale) for k, v in self.emission[label].items()}
            self.transition[label] = {k: (v/scale) for k, v in self.transition[label].items()}"""


    def dump_emissions(self) -> dict:
        return self.emission


    def dump_transitions(self) -> dict:
        return self.transition


    def viterbi(self, feature_seq: list) -> tuple:
        viterbi = np.zeros((len(self.labels), len(feature_seq)))
        backpointer = np.zeros((len(self.labels), len(feature_seq)))
        for s in range(len(self.labels)):
            emission = self.dot(self.emission[self.labels[s]],feature_seq[0])
            viterbi[s,0] = self.initial[s] + emission
            backpointer[s,0] = 0
        for t in range(1, len(feature_seq)):
            for s in range(len(self.labels)):
                emission = self.dot(self.emission[self.labels[s]], feature_seq[t])
                max = -np.inf
                for s_prime in range(len(self.labels)):
                    transition = self.transition[self.labels[s_prime]][self.labels[s]]
                    score = viterbi[s_prime,t-1] + transition + emission
                    if score > max:
                        max = score
                        viterbi[s,t] = max
                        backpointer[s,t] = s_prime
        terminate = np.argmax(viterbi[:,len(feature_seq)-1])
        best_path = []
        for t in reversed(range(len(feature_seq))):
            best_path.append(self.labels[int(terminate)])
            terminate = backpointer[int(terminate), t]
        return tuple(reversed(best_path))


    # dot product of feature dictionary and weight dictionary
    def dot(self, param: dict, feature_dict: dict) -> int:
        dot = 0
        for feat in feature_dict:
            if feat in self.feature_set:
                dot += param[feat]*feature_dict[feat]
        return dot


    # updates the transition and emission weights associated to the incorrect index
    def update(self, predicted: list, gold: list, sentence_features: dict, index: int) -> None:
        pred_tag = predicted[index]
        gold_tag = gold[index]
        for i,feat in enumerate(sentence_features[index]): #update emissions
            self.emission[pred_tag][feat] -= 1
            self.emission[gold_tag][feat] += 1
        if index == 0:  #update initial
            self.initial[pred_tag] -= 1
            self.initial[gold_tag] += 1
        if index > 0:   #update previous transitions
            prev_pred = predicted[index-1]
            prev_gold = gold[index-1]
            self.transition[prev_pred][pred_tag] -= 1
            self.transition[prev_gold][gold_tag] += 1
        if index < len(sentence_features)-1: #update next transition
            next_pred = predicted[index+1]
            next_gold = gold[index+1]
            self.transition[pred_tag][next_pred] -= 1
            self.transition[gold_tag][next_gold] += 1


    #evaluation: accuracy, precision, recall, f1
    def predict(self, data: Corpus) -> None:
        tp = 0
        fp = 0
        fn = 0
        hits = 0
        total = 0
        for i, doc in enumerate(data):
            sentences = doc.sentences
            labels = doc.labels
            for j,sent in enumerate(sentences):
                gold = labels[j]
                sent = extract_sentence_features(sent)
                predicted = self.viterbi(sent)
                gold_span = self.extract_entity(gold)
                pred_span = self.extract_entity(predicted)
                for entity in pred_span:
                    if entity in gold_span:
                        tp +=1
                    else:
                        fp +=1
                for entity in gold_span:
                    if entity not in pred_span:
                        fn +=1
                for k,label in enumerate(predicted):
                    total +=1
                    if label == gold[k]:
                        hits +=1
        #don't average
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        accuracy = hits/total
        f1 = (2*precision*recall) / (precision+recall)
        print("accuracy =  " + str(accuracy))
        print("precision = " + str(precision))
        print("recall = " + str(recall))
        print("f1 = " + str(f1))
        print()


    def _get_spans(self, label_seq: list) -> list:
        labels = []
        for i, label in enumerate(label_seq):
            if label!='O':
                split_label = label.split('-')
                split_label.append(i)
                labels.append(tuple(split_label))
        labels.append((None, None, -1))
        return labels


    #sample:
    #'B-MISC', 'B-PER', 'I-PER', 'O', 'O', 'O'
    #'I-MISC', 'B-PER', 'B-PER', 'O', 'O', 'O'
    def extract_entity(self, label: list) -> list:
        spans = self._get_spans(label)
        total_spans = []
        for i, (ent_class, ent_type, index) in enumerate(spans):
            if i == 0:
                span_lst=[]
                span_lst.append(spans[i])
            else:
                prev_class, prev_type, prev_index = spans[i-1]
                if prev_index != index-1 or ent_class == 'B':
                    total_spans.append(span_lst)
                    span_lst = []
                    span_lst.append(spans[i])
                else:
                    if ent_type == prev_type:
                        span_lst.append(spans[i])
                    else:
                        total_spans.append(span_lst)
                        span_lst = []
                        span_lst.append(spans[i])
        return total_spans


if __name__ == "__main__":
    train_path= "data/conll2003/en/train.txt"
    val_path= "data/conll2003/en/valid.txt"
    test_path= "data/conll2003/en/test.txt"
    train_data= load_conll(train_path)
    val_data= load_conll(val_path)
    test_data= load_conll(test_path)
    for i in range(20):
        print("total number of epochs = " + str(i+1))
        perceptron= Structured_Perceptron(epoch=i+1)
        perceptron.train(train_data)
        perceptron.predict(val_data)


