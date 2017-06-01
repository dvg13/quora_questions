from spacy.en import English
import numpy as np
import pandas as pd
from queue import Queue
from copy import copy

class Reader():
    def __init__(self,questions_file,val_size=0):
        self.qdf = pd.read_csv(questions_file)
        self.val_size = val_size
        print("Loaded questions...")

        #split into train and validation
        idx = np.arange(len(self.qdf.index))
        np.random.shuffle(idx)
        self.val_idx = idx[:self.val_size]
        self.train_idx = idx[self.val_size:]

        self.pos_idx = self.qdf.ix[self.train_idx][self.qdf.ix[self.train_idx]['is_duplicate'] == 1].index.tolist()
        self.neg_idx = self.qdf.ix[self.train_idx][self.qdf.ix[self.train_idx]['is_duplicate'] == 0].index.tolist()

        self.pos_idx_idx = 0
        self.neg_idx_idx = 0
        self.val_idx_idx = 0
        self.val_epoch = 0

        #load parser
        self.parser = English()
        print("Loaded parser...")
        self.end_token = "^"

        #keep stats of how many words have/don't have embeddings
        self.has_vector = 0
        self.no_vector = 0

    def get_q_vector(self,question,add_end_token=False):
        if add_end_token:
            question += self.end_token

        q1_vector = []
        parsed = self.parser(question)

        for token in parsed:
            if token.has_vector:
                self.has_vector += 1
            else:
                self.no_vector += 1
            q1_vector.append(token.vector)

        return np.stack(q1_vector)

    def combine_vectors(self,vectors):
        max_words = 0
        for vector in vectors:
            if vector.shape[0] > max_words:
                max_words = vector.shape[0]
        for i in range(len(vectors)):
            empty_words = max_words - vectors[i].shape[0]
            vectors[i] = np.vstack((vectors[i],np.zeros((empty_words,vectors[i].shape[1]))))
        return np.stack(vectors)

    def get_row(self,idx_list,idx_list_idx):
        if idx_list_idx == len(idx_list):
            np.random.shuffle(idx_list)
            idx_list_idx = 0
        row = idx_list[idx_list_idx]

        return row,idx_list_idx+1

    def process_row(self,row,q1_vectors,q2_vectors,targets,add_end_token):
        try:
            q1_vectors.append(self.get_q_vector(self.qdf.ix[row]["question1"],add_end_token))
            q2_vectors.append(self.get_q_vector(self.qdf.ix[row]["question2"]))
            targets.append(self.qdf.ix[row]["is_duplicate"])
        except Exception as e:
            print(e)

    #can only use even batch_sizes
    def next(self,batch_size,add_end_token=False):
        q1_vectors = []
        q2_vectors = []
        targets = []

        while len(q1_vectors) < batch_size:
            #get negative row
            neg_row,self.neg_idx_idx = self.get_row(self.neg_idx,self.neg_idx_idx)
            pos_row,self.pos_idx_idx = self.get_row(self.pos_idx,self.pos_idx_idx)

            self.process_row(neg_row,q1_vectors,q2_vectors,targets,add_end_token)
            self.process_row(pos_row,q1_vectors,q2_vectors,targets,add_end_token)

        #combine the above into numpy vectors
        return self.combine_vectors(q1_vectors),self.combine_vectors(q2_vectors),np.stack(targets)

    def next_val(self,batch_size,add_end_token=False):
        q1_vectors = []
        q2_vectors = []
        targets = []

        while len(q1_vectors) < batch_size and self.val_idx_idx < len(self.val_idx):
            val_row,self.val_idx_idx = self.get_row(self.val_idx,self.val_idx_idx)
            self.process_row(val_row,q1_vectors,q2_vectors,targets,add_end_token)

        if self.val_idx_idx == len(self.val_idx):
            np.random.shuffle(self.val_idx)
            self.val_idx_idx = 0
            self.val_epoch += 1

        #combine the above into numpy vectors
        return self.combine_vectors(q1_vectors),self.combine_vectors(q2_vectors),np.stack(targets)

if __name__ == "__main__":
    reader = Reader("train.csv",10000)
    print(len(reader.val_idx) + len(reader.pos_idx) + len(reader.neg_idx))

    reader = Reader("train.csv",20000)
    print(len(reader.val_idx) + len(reader.pos_idx) + len(reader.neg_idx))
