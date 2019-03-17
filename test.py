import  tensorflow as tf
from utils import  TextTransform
from model import CharRNN
import  os


tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/converter.pkl', 'converter output file path')
tf.flags.DEFINE_string('checkpoint_path', 'model/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 300, 'max length to generate')


FLAGS = tf.flags.FLAGS


def main(_):

    converter = TextTransform(output_file_path=FLAGS.converter_path)

    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =tf.train.latest_checkpoint(FLAGS.checkpoint_path)


    model = CharRNN(converter.vocab_size, sampling=True,
                        lstm_size=FLAGS.lstm_size,
                        num_layers=FLAGS.num_layers,
                        use_embedding=FLAGS.use_embedding,
                        embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)


    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()


