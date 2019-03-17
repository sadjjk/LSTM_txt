import collections
import numpy as np
import pickle
import jieba
import copy


def batch_generate(arr,n_seqs,n_steps,epoch_size = 10):

    arr = copy.deepcopy(arr)


    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape(n_seqs,-1)

    # print(arr.shape)
    # print(len(range(0,arr.shape[1],n_steps)))

    print("epoch_size: {} total_steps:{}".format(epoch_size,epoch_size*len(range(0,arr.shape[1],n_steps))))

    for j in range(epoch_size):
        np.random.shuffle(arr)
        for i in range(0,arr.shape[1],n_steps):
            x = arr[:,i:i+n_steps]
            y = np.zeros_like(x)
            y[:,:-1],y[:,-1] = x[:,1:],x[:,0]
            yield  x,y



class TextTransform():
    def __init__(self,input_file_path =None ,max_vocab_len = 10000,
                      output_file_path = None,min_particle_size = 'word'):

        if output_file_path is not None:
            with open(output_file_path,'rb') as f :
                self.vocab = pickle.load(f)

        else:
            with open(input_file_path,encoding='utf-8') as f :

                assert min_particle_size  in ('word','phrase'), "min_particle_size must be word or phrase"

                if min_particle_size == 'word':
                    self.text = list(f.read())
                else :
                    self.text = jieba.cut(f.read())

                count_pairs = sorted(collections.Counter(self.text).items(),key = lambda x:-x[1])
                vocab,_ = list(zip(*count_pairs))

                if len(vocab) >= max_vocab_len:
                    vocab = vocab[:max_vocab_len]

                self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):

        return len(self.vocab)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)




# aa = TextTransform('data/shakespeare.txt')
# arr = aa.text_to_arr(aa.text)
# g = batch_generate(arr, 100,100)
# print(g)


