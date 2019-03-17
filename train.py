import os

import tensorflow as tf
from utils import TextTransform, batch_generate
from model import CharRNN
tf.flags.DEFINE_string('model_name', 'model', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('log_every_n', 100, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 35000, 'max char number')
tf.flags.DEFINE_string('min_particle_size', 'word', 'min particle size')
tf.flags.DEFINE_integer('epoch_size', 10, 'epoch size')

FLAGS = tf.flags.FLAGS


def main(_):
    if os.path.exists(FLAGS.model_name) is False:
        os.mkdir(FLAGS.model_name)

    converter = TextTransform(FLAGS.input_file, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(FLAGS.model_name, 'converter.pkl'))

    arr = converter.text_to_arr(converter.text)

    g = batch_generate(arr, FLAGS.num_seqs, FLAGS.num_steps, FLAGS.epoch_size)

    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )

    model.train(g,
                FLAGS.model_name,
                FLAGS.log_every_n)


if __name__ == "__main__":
    tf.app.run()
