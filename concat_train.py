import tensorflow as tf
import numpy as np
import balanced_reader as reader
import concat_model as model
import argparse
import sys

#there has to be a better way to do this
def get_summary_from_scalar(name,scalar):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=scalar)])

def train():
    qreader = reader.Reader(FLAGS.qfile,FLAGS.val_size)
    qmodel = model.ConcatPairClassifier(FLAGS.batch_size,FLAGS.hidden_size)

    q1_placeholder = tf.placeholder(tf.float32,[FLAGS.batch_size,None,FLAGS.embedding_size])
    q2_placeholder = tf.placeholder(tf.float32,[FLAGS.batch_size,None,FLAGS.embedding_size])
    target_placeholder = tf.placeholder(tf.float32,[FLAGS.batch_size])

    loss,pred = qmodel.model(q1_placeholder,q2_placeholder,target_placeholder)
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)

    # Compute the gradients for a list of variables.
    grads = optimizer.compute_gradients(loss)
    clipped_grads = [(tf.clip_by_value(grad,-1,1),var) for grad,var in grads]
    train_op = optimizer.apply_gradients(clipped_grads)

    summary_fetches = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    #for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #    print(var)
    #pause = input()

    with tf.Session() as session:
        session.run(init)

        summary_writer = tf.summary.FileWriter("summaries",session.graph)

        loss_sum,accuracy_sum=0,0

        for i in range(FLAGS.max_steps):
            q1,q2,target = qreader.next(FLAGS.batch_size,True)

            try:
                feed_dict = {q1_placeholder: q1,
                             q2_placeholder: q2,
                             target_placeholder: target}

                _,loss_val,predicted = session.run([train_op,loss,pred],feed_dict=feed_dict)
                predicted = np.round(predicted)
                accuracy = np.sum(predicted == target) / target.shape[0]

                loss_sum+=loss_val/(len(predicted)*100)
                accuracy_sum += accuracy/100

                if (i +1) % 100 == 0:
                    print(i,loss_sum,accuracy_sum,np.mean(predicted))
                    loss_sum,accuracy_sum=0,0
                    summaries = session.run([summary_fetches],feed_dict=feed_dict)
                    for s in summaries:
                        summary_writer.add_summary(s,i)

                if (i + 1) % 5000 == 0 and FLAGS.val_size > 0:
                    #get validation loss
                    starting_epoch = qreader.val_epoch
                    validation_loss = 0
                    validation_accuracy = 0

                    while qreader.val_epoch == starting_epoch:
                        q1,q2,target = qreader.next_val(FLAGS.batch_size,True)
                        try:
                            feed_dict = {q1_placeholder: q1,
                                        q2_placeholder: q2,
                                        target_placeholder: target}

                            loss_val,predicted = session.run([loss,pred],feed_dict=feed_dict)
                            predicted = np.round(predicted)
                            accuracy = np.sum(predicted == target)
                            validation_loss += loss_val / len(qreader.val_idx)
                            validation_accuracy += accuracy / len(qreader.val_idx)
                        except Exception as e:
                            print(e)

                    #add a summary and print
                    validation_loss_summary = get_summary_from_scalar('validation_loss',validation_loss)
                    summary_writer.add_summary(validation_loss_summary,i)
                    validation_acc_summary = get_summary_from_scalar('validation_accuracy',validation_accuracy)
                    summary_writer.add_summary(validation_acc_summary,i)
                    print("Validation",validation_loss,validation_accuracy)


                if (i+1) % 10000== 0:
                    saver.save(session,"question_model",global_step=i)

            except Exception as e:
                print(e)
                #:)

def main():
    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--qfile',
        type=str,
        default="train.csv",
        help='location of questions csv'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=300,
        help='size of word embeddings'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='size of hidden/cell state for LSTM'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning Rate'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=32000,
        help='Size of Validation Set'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
