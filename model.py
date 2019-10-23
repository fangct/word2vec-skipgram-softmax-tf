import tensorflow as tf
import numpy as np

class Skipgram(object):

    def __init__(self, args, vocab_size):
        self.vocab_size = vocab_size
        self.embedding_dim = args.embedding_dim
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.window_size = args.window_size

    def add_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.vocab_size], name='input_x')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.vocab_size], name='input_x')

    def project_layer(self):
        self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
        WT = tf.Variable(tf.random_uniform([self.embedding_dim, self.vocab_size], -1.0, 1.0))

        hidden = tf.matmul(self.inputs, self.W)  # [batch_size, embedding_size]
        self.output = tf.matmul(hidden, WT) # [batch_size, voc_size]


    def loss_layer(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output)
        self.loss = tf.reduce_mean(loss)

    def train_op(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)


    def build_model(self):
        self.add_placeholder()
        self.project_layer()
        self.loss_layer()
        self.train_op()


    def batch_iter(self, data, batch_size):
        np.random.shuffle(data)
        batch_inputs, batch_labels = [], []
        for one_data in data:
            if len(batch_inputs) == batch_size:
                yield (batch_inputs,batch_labels)
                batch_inputs, batch_labels = [], []

            batch_inputs.append(np.eye(self.vocab_size)[one_data[0]])
            batch_labels.append(np.eye(self.vocab_size)[one_data[1]])

        if len(batch_inputs) != 0:
            yield (batch_inputs,batch_labels)


    def train(self, data):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(self.num_epochs):
                num_batches = (len(data) + self.batch_size + 1) // self.batch_size
                batch_data = self.batch_iter(data, self.batch_size)

                for step, (batch_inputs, batch_labels) in enumerate(batch_data):
                    _, loss, global_step = sess.run([self.train_step, self.loss, self.global_step], feed_dict={self.inputs: batch_inputs, self.labels: batch_labels})

                    if (step + 1) % 10 == 0 or (step +1) == 1 or (step +1) == num_batches:
                        print('Epoch: {}, step: {}, loss: {:.6f}, global_step: {}'.format(epoch +1, step+1, loss, global_step))

            trained_embeddings = self.W.eval()

        return trained_embeddings