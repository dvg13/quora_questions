import tensorflow as tf
import numpy as np
import test_reader as reader
import concat_model as model
import argparse
import sys

def test():
    qreader = reader.Reader(FLAGS.qfile)

    #create placeholders
    q1_placeholder = tf.placeholder(tf.float32,[FLAGS.batch_size,None,FLAGS.embedding_size])
    q2_placeholder = tf.placeholder(tf.float32,[FLAGS.batch_size,None,FLAGS.embedding_size])

    qmodel = model.ConcatPairClassifier(FLAGS.batch_size,FLAGS.hidden_size)
    _,pred = qmodel.model(q1_placeholder,q2_placeholder,tf.zeros([FLAGS.batch_size]))

    #open the text file to read this to
    output_file = open(FLAGS.output_file,'w')
    output_file.write("test_id,is_duplicate\n")

    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session, FLAGS.model)

        while not qreader.finished:
            #need the reader to give me no target
            q1,q2,qids = qreader.next(FLAGS.batch_size,True)

            try:
                feed_dict = {q1_placeholder: q1,
                             q2_placeholder: q2}

                predicted = session.run(pred,feed_dict=feed_dict)

                for i in range(len(qids)):
                    output_file.write("{},{}\n".format(qids[i],predicted[i]))

            except Exception as e:
                print(e)
                #:)

    output_file.close()

def main():
    test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--qfile',
        type=str,
        default="test.csv",
        help='location of questions csv'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=612,
        help='Batch size'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=300,
        help='size of word embeddings'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default='questions_test.txt',
        help='output file for predictions'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='',
        help='ckpt file for model'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='size of hidden/cell state for LSTM'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
