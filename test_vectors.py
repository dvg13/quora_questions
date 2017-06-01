import pandas as pd
from spacy.en import English

def test_questions(questions_file):
    parser = English()
    print("loaded parser")

    questions = pd.read_csv(questions_file)
    print("loaded questions")

    #unmatched = open("unmatched", 'w')

    for i in range(len(questions.index)):
        if i % 1000 == 0:
            print(i)

        row = questions.ix[i]

        try:
            sentence1 = parser(row["question1"])
            for token in sentence1:
                if len(token.vector) != 300:
                    print(token)
                    pause = input()
        except:
            print(row)
            pause = input()

        try:
            sentence2 = parser(row["question2"])
            for token in sentence2:
                if len(token.vector) != 300:
                    print(token)
                    pause = input()
        except:
            print(row)
            pause = input()

test_questions("train.csv")
