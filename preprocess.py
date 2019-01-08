import numpy as np 
from pdf2txt import file_names
from collections import Counter
import pickle
from keras.preprocessing.sequence import pad_sequences
import os
from pretag import fenci, total_stop_words


def parse_file(file_dir):
    '''
    将file_dir转变成[[[word, tag], [word, tag], [word, tag]],...]的形式
    '''
    split_text = '\n'
    with open(file_dir, 'r', encoding='utf-8') as f:
        string = f.read()
        data = [[word_tag.split() 
            for word_tag in sentence.split(split_text)] 
            for sentence in string.strip().split(split_text + split_text)]
        '''
        remove_list = []

        for sentence in data:
            if len(sentence) > 100: 
                remove_list.append(sentence)
        
        for item in remove_list:
            data.remove(item)
        '''
        return data

def parse_folder(folder_dir):
    '''
    将整个文件夹中的文件都转化为[[[word, tag], [word, tag], [word, tag]],...]的形式
    '''
    root, files = file_names(folder_dir)
    output = []
    for file in files:
        input_file = root + '/' + file
        data = parse_file(input_file)
        output += data
    return output

def load_data():
    '''
    '''
    train_dir = 'C:/Users/lisicong/Desktop/shouxian/tag/train'
    test_dir = 'C:/Users/lisicong/Desktop/shouxian/tag/test'
    train = parse_folder(train_dir)
    test = parse_folder(test_dir)
    word_counts = Counter(word_tag[0].lower() for sentence in train for word_tag in sentence)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    vocab.insert(0, '<unk>')
    vocab.insert(0, '<padding>')
    tags = ['O']

    with open('C:/Users/lisicong/Desktop/shouxian/model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, tags), outp)

    train = process_data(train, vocab, tags)
    test = process_data(test, vocab, tags)
    return train, test, (vocab, tags)

def process_data(data, vocab, tags):
    '''
    '''
    #maxlen = max(len(s) for s in data)
    maxlen = 100
    word2idx = dict((w, i) for i, w in enumerate(vocab)) #将每一个单词的频率表示出来，words，frequence
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  
    y = [[tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen)
    y = pad_sequences(y, maxlen, value=-1)
    y = np.expand_dims(y, 2)
    return x, y

def process(file_dir, vocab, maxlen=100):
    '''
    将未标注的文件转为word_idx的形式，用于预测
    默认该文件已经由PDF格式转化为txt的格式
    且该格式和pdf2txt中的转化是一样的
    '''
    stop_words = total_stop_words()
    text = fenci(file_dir, stop_words)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    '''
    temp = []
    for sentence in text:
        tmp = []
        for word in sentence:
            print(word)
            tmp.append(word2idx.get(word.lower(), 1))
        temp.append(tmp)
    text = temp
    '''
    text = [[word2idx.get(word.lower(), 1) for word in sentence] for sentence in text]
    text = pad_sequences(text, maxlen=maxlen)
    return text


    


if __name__ == '__main__':
    #file_dir = 'tag/train/116母婴安康定期寿险.txt'1
    #data = parse_file(file_dir)
    #print(data)

    load_data()





