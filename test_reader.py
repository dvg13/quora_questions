from spacy.en import English
import numpy as np
import pandas as pd
from queue import Queue
from copy import copy

class Reader():
    def __init__(self,questions_file,val_size=0):
        self.qdf = pd.read_csv(questions_file)
        self.n = len(self.qdf.index)
        self.idx = 0
        self.finished = False
        print("Loaded questions...")

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

    def process_row(self,row,q1_vectors,q2_vectors,qids,add_end_token):
        try:
            question_1 = self.get_q_vector(row["question1"],add_end_token)
            question_2 = self.get_q_vector(row["question2"])

        except Exception as e:
            question_1 = np.zeros([1,300])
            question_2 = np.zeros([1,300])

        qids.append(row["test_id"])
        q1_vectors.append(question_1)
        q2_vectors.append(question_2)

    #assume that batch size goes evenly into test size
    #can always use one
    def next(self,batch_size,add_end_token=False):
        qids = []
        q1_vectors = []
        q2_vectors = []

        while len(q1_vectors) < batch_size:
            #row
            row = self.qdf.ix[self.idx]
            self.process_row(row,q1_vectors,q2_vectors,qids,add_end_token)
            self.idx += 1

        if self.idx == self.n:
            self.finished = True

        #combine the above into numpy vectors
        return self.combine_vectors(q1_vectors),self.combine_vectors(q2_vectors),qids

if __name__ == "__main__":
    reader = Reader("test.csv")
    result = reader.next(612,True)
    print(result[0].shape,result[1].shape,len(result[2]))
