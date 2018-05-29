import logging
import os
import torch
from torchtext import data
import numpy as np
import logging.config

from cove.encoder import MTLSTM

def get_cove_sentence_embeddings(source_dir, train_file, valid_file=None, test_file=None, batch_size=32, gpu=-1):
    logger = logging.getLogger('cove')
    hdlr = logging.FileHandler(os.path.join(source_dir, 'cove.log'))
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    logger.info('encode {} to cove'.format(source_dir))

    text = data.Field(lower=True, include_lengths=True, batch_first=True)

    sentence_files = []
    sentences = []
    train_file = os.path.join(source_dir, train_file)
    with open(train_file, 'r') as f:
        train = list(map(lambda x: data.Example.fromlist([x.strip()], fields=[('sentence', text)]), f.readlines()))
        train = data.Dataset(train, fields=[('sentence', text)])

    logger.info('Building vocabulary')
    text.build_vocab(train)
    text.vocab.load_vectors("glove.840B.300d")

    cove = MTLSTM(n_vocab=len(text.vocab), vectors=text.vocab.vectors)
    if gpu == -1 or gpu is None:
        cove.cpu()
        device = None
    else:
        cove.cuda(gpu)
        device = gpu

    train_iter = data.Iterator(train, batch_size=batch_size, train=False, shuffle=False, repeat=False, sort=False, device=device)

    if valid_file:
        valid_file = os.path.join(source_dir, valid_file)
        with open(valid_file, 'r') as f:
            valid = list(map(lambda x: data.Example.fromlist([x.strip()], fields=[('sentence', text)]), f.readlines()))
            valid = data.Dataset(valid, fields=[('sentence', text)])
            valid_iter = data.Iterator(valid, batch_size=batch_size, train=False, shuffle=False, repeat=False, sort=False, device=device)
    else:
        valid = None
        valid_iter = None
    if test_file:
        test_file = os.path.join(source_dir, test_file)
        with open(test_file, 'r') as f:
            test = list(map(lambda x: data.Example.fromlist([x.strip()], fields=[('sentence', text)]), f.readlines()))
            test = data.Dataset(test, fields=[('sentence', text)])
            test_iter = data.Iterator(test, batch_size=batch_size, train=False, shuffle=False, repeat=False, sort=False, device=device)
    else:
        test = None
        test_iter = None

    for i, iter_file in enumerate(zip([train_iter, valid_iter, test_iter], [train_file, valid_file, test_file])):
        if iter_file[0] is None:
            continue
        logger.info('Generating CoVe for %s' % (iter_file[1]))
        sentences = []
        for batch_idx, batch in enumerate(iter_file[0]):
            cove.eval()
            inputs, lengths = batch.sentence
            cove_sentence = cove(inputs, lengths)
            cove_sentence = cove_sentence.data.cpu()
            for j in range(cove_sentence.shape[0]):
                sentences.append(cove_sentence[j][lengths[j]-1].numpy())
            if (batch_idx % 100) == 0:
                logger.info('%d...' % (batch_idx * batch_size))
        sentences = np.array(sentences, dtype=object)
        npy_file = iter_file[1].replace('.txt', '_embedding.npy')
        np.save(npy_file, sentences)


if __name__ == '__main__':
    source_dir = 'data'
    file_names = ['train.txt', 'valid.txt', 'test.txt']
    gpu = -1
    get_cove_sentence_embeddings(source_dir, *file_names, gpu=gpu)
