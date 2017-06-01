import tensorflow as tf

class ConcatPairClassifier():
    def __init__(self,batch_size,hidden_size):
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def lstm(self,sentence):
        """
        input is a zero-padded b * words * vector length matrix of word vectors
        return the last output/hidden state
        """
        #lstm_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        output, state = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)

        return tf.reshape(tf.slice(output,[0,tf.shape(output)[1]-1,0],[self.batch_size,1,self.hidden_size]),
                         [self.batch_size,self.hidden_size])

    def classifier(self,hidden):
        """
        input is a vector of length self.hidden_size * 2
        output is a binary target
        """
        w = tf.Variable(tf.truncated_normal([self.hidden_size], stddev=0.02),name="weights")
        b = tf.Variable(tf.zeros([1]),name="bias")
        return tf.sigmoid(tf.reduce_sum(tf.multiply(w,hidden),axis=1) + b)

    def model(self,q1,q2,target):
        """
        input is two sentences with shape batch * words * word_vector_size
        the sentences do not have to have the same number of words
        target is a binary array of shape batch
        """

        #B,Words,vector size,
        concatenated = tf.concat([q1,q2],axis=1)

        with tf.variable_scope("LSTM") as scope:
            h = self.lstm(concatenated)
            tf.summary.histogram('h', h)


        with tf.variable_scope("CL") as scope:
            pred = self.classifier(h)
            tf.summary.histogram('pred', pred)

        loss = target * tf.log(1e-8 + pred) + (1 - target) * tf.log(1 - pred + 1e-8)
        tf.summary.scalar('loss',tf.reduce_mean(-loss))

        return tf.reduce_sum(-loss), pred
