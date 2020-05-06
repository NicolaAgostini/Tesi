import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tqdm
import re
from functools import reduce
import math
from Main import root_path


#path_dsk = "/Users/nicolago/Desktop/"

class Glove():
    def __init__(self, path_glove, alpha, which_methods = "", temperature = 0):
        """
        :param path_glove:
        :param alpha:
        :param which_methods: softmax, standard or nothing for label smoothing with prior
        :param temperature: parameter T of the softmax
        """

        self.path_glove = path_glove
        self.alpha = alpha
        self.which_methods = which_methods
        self.temperature = temperature
        self.embeddings_dict = {}
        self.phi = np.array(0)
        self.Pi = np.array(0)
        self.y = np.array(0)
        self.y_soft = np.array(0)


    def load_glove(self):
        """
        load the GloVe file
        """
        with open(self.path_glove+"glove.6B.50d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector

    def find_similar(self, target):
        return sorted(self.embeddings_dict.keys(),
                      key=lambda word: spatial.distance.euclidean(self.embeddings_dict[word], self.embeddings_dict[target]))

    def compute_phi(self):
        """
        produce a list of values for phi function that are concat of verb and noun vectors, if are list take avg vectors
         between them
        """
        actions = self.return_all_actions()
        phi = []
        for key, values in actions.items():
            concat = 0
            #print(values)
            verbs = values[0]  # verb può essere anche una lista di verbi
            nouns = values[1]
            verbs_np = []
            nouns_np = []
            for v in verbs:
                verbs_np.append(self.embeddings_dict[v.lower()])  # nouns e verbs sono liste di array numpy
            for n in nouns:
                nouns_np.append(self.embeddings_dict[n.lower()])
            verbs = np.array(verbs_np)
            nouns = np.array(nouns_np)

            verbs = np.mean(verbs, axis=0)
            nouns = np.mean(nouns, axis=0)

            #print(verb)
            concat = np.concatenate((verbs, nouns))
            phi.append(concat)

        self.phi = np.array(phi)





    def return_all_actions(self):
        """
        :return: actions: a dictionary with index k, for class k belongs to {1,106}, and values actions[k]= tuple
        ([verb1,verb2,...], [noun1,noun2,..])
        """
        actions = {}
        with open(root_path + "action_annotation/action_idx.txt", 'r') as f:
            for line in f:
                line = line.strip()
                values = re.split("/| |_|,",line)

                verbs = []
                nouns = []


                for v in values[:-1]:
                    #print(v)
                    if v[0].isupper():  # then it is a verb
                        verbs.append(v)
                    else:
                        nouns.append(v)

                actions[values[-1]] = (verbs, nouns)
        return actions

    def compute_Pi(self):
        """
        :param temperature:
        Pi[k,i] which is the absolute value of scalar producto between phi(k) and phi(i) divided by
            the summation over all j of the abs value of scalar product between phi(k) and phi(j)
        """
        n_dim = self.phi.shape[0]  # is the number of labels in my classification problem
        Pi = np.zeros((n_dim, n_dim))

        if self.which_methods == "softmax":
            for k in range(n_dim):
                for i in range(n_dim):
                    numerator = abs(np.dot(self.phi[k], self.phi[i]))
                    temp = 0
                    for j in range(n_dim):
                        temp += math.exp(abs(np.dot(self.phi[k], self.phi[j]))*1/self.temperature)
                    denominator = temp
                    Pi[k][i] = math.exp(numerator*1/self.temperature) / denominator

        else:
            for k in range(n_dim):
                for i in range(n_dim):
                    numerator = abs(np.dot(self.phi[k], self.phi[i]))
                    temp = 0
                    for j in range(n_dim):
                        temp += abs(np.dot(self.phi[k], self.phi[j]))
                    denominator = temp
                    Pi[k][i] = numerator / denominator

        self.Pi = np.array(Pi)

    def print_heatmap(self):
        """
        print the heatmap of Pi matrix
        """
        plt.imshow(self.Pi, cmap='hot', interpolation='nearest')
        plt.show()


    def compute_onehot(self):
        """
        generate a one hot encoder y of shape (106,106) where each row is a class label of all zeros except k-th component
            which 1 for k-th class
        """
        n_dim = self.Pi.shape[0]
        y = np.zeros((n_dim, n_dim))
        with open(root_path + "action_annotation/action_idx.txt", 'r') as f:
            for line in f:
                line = line.strip()
                values = re.split("/| |_|,", line)
                y[int(values[-1])-1][int(values[-1])-1] = 1

        self.y = np.array(y)

    def get_onehot(self):
        """
        :return: one hot encoder
        """
        self.compute_onehot()
        return self.y

    def compute_ysoft(self):
        """
        compute the y_soft, first computing phi then Pi, y-one hot and last computing y_soft
        """
        self.load_glove()
        self.compute_phi()
        self.compute_Pi()
        self.compute_onehot()
        y_soft = []
        n_labels = self.y.shape[0]
        if self.which_methods == "standard":
            for i in range(n_labels):
                y_soft.append((1-self.alpha)*self.y[i] + self.alpha*(1/n_labels))
        else:
            for i in range(n_labels):
                temp = (1-self.alpha)*self.y[i] + self.alpha*self.Pi[i]
                y_soft.append(temp)

        self.y_soft = np.array(y_soft)

    def get_ysoft(self):
        """
        :return: y_soft array of smoothed labels
        """
        self.compute_ysoft()
        return self.y_soft

