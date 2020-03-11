import os
import utils
import numpy as np
import gensim.models.keyedvectors as word2vec

from tflearn.data_utils import pad_sequences


def data_word2vec_one_label(input_file,
                            word2vec_model):
    """
    Create the research data token index based on the word2vec model file.
    Return the class Data(includes the data token index and data labels).

    Args:
        input_file: The research data
        word2vec_model: The word2vec model file
    Returns:
        The class Data(includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    # if not input_file.endswith('.json'):
    #     raise IOError("✘ The research data is not a json file. "
    #                   "Please preprocess the research data into the "
    #                   "json file.")

    raw_tokens_list_gov = []
    raw_tokens_list_art = []
    test_id_list = []
    content_index_list_gov = []
    content_index_list_art = []
    # labels_list = []
    onehot_labels_list = []
    labels_num_list = []
    total_line = 0

    data = utils.load_pickle(input_file)
    test_id = data['testid']
    features_content_gov = data['gov']
    features_content_art = data['art']
    label = data['label']

    test_id_list.append(test_id)
    content_index_list_gov.append(_token_to_index(
        features_content_gov))
    content_index_list_art.append(_token_to_index(
        features_content_art))

    raw_tokens_list_gov.append(features_content_gov)
    raw_tokens_list_art.append(features_content_art)

    onehot_labels_list.append(label)
    labels_num = 1
    labels_num_list.append(labels_num)
    total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def testid(self):
            return test_id_list

        @property
        def raw_tokens_gov(self):
            return raw_tokens_list_gov

        @property
        def raw_tokens_art(self):
            return raw_tokens_list_art

        @property
        def tokenindex_gov(self):
            return content_index_list_gov

        @property
        def tokenindex_art(self):
            return content_index_list_art

        @property
        def onehot_labels(self):
            return onehot_labels_list

        @property
        def labels_num(self):
            return labels_num_list

    return _Data()


def load_data_and_labels_one_label(data_file,
                                   word2vec_path,
                                   use_pretrain=True):
    """
    Load research data from files, splits the data into words and generates
    labels. Return split sentences, labels and the max sentence length of
    the research data.

    Args:
        data_file: The research data
        num_labels: The number of classes
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
        word2vec_path: path of pretrained word2vec
        use_pretrain: whether to use pretrained word2vec
    Returns:
        The class Data
    """

    ###########################################################################
    #
    # word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'
    #
    # # Load word2vec model file
    # if not os.path.isfile(word2vec_file):
    #     create_word2vec_model(embedding_size, TEXT_DIR)
    ###########################################################################

    # word2vec_path = '../../Word2Vec/GoogleNews-vectors-negative300.bin'

    if use_pretrain:
        model = word2vec.KeyedVectors.load_word2vec_format(word2vec_path,
                                                           binary=True,
                                                           limit=250000)
    else:
        print("only supports use_pretrain mode ")

    # Load data from files and split by words
    data = data_word2vec_one_label(input_file=data_file,
                                   word2vec_model=model)
    return data


def pad_data_one_label(data,
                       pad_seq_len_gov,
                       pad_seq_len_art):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        pad_seq: The padded data
        labels: The data labels
    """
    pad_seq_gov = pad_sequences(data.tokenindex_gov,
                                maxlen=pad_seq_len_gov,
                                value=0.)

    pad_seq_art = pad_sequences(data.tokenindex_art,
                                maxlen=pad_seq_len_art,
                                value=0.)

    onehot_labels = data.onehot_labels
    return pad_seq_gov, \
           pad_seq_art, \
           onehot_labels


def load_vocab_size(word2vec_path):
    """
    Return the vocab size of the word2vec file.

    Args:
        embedding_size: The embedding size
        word2vec_path: Path of word2vec
    Returns:
        The vocab size of the word2vec file
    Raises:
        IOError: If word2vec model file doesn't exist
    """

    if not os.path.isfile(word2vec_path):
        raise IOError("✘ The word2vec file doesn't exist."
                      "Please use function <create_vocab_"
                      "size(embedding_size)> to create it!")

    model = word2vec.KeyedVectors.load_word2vec_format(
        word2vec_path, binary=True, limit=400000)

    return len(model.wv.vocab.items())


def load_word2vec_matrix(vocab_size,
                         embedding_size,
                         word2vec_path):
    """
    Return the word2vec model matrix.

    Args:
        vocab_size: The vocab size of the word2vec model file
        embedding_size: The embedding size
        word2vec_path: path of pretrained word2vec
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist

    """
    if not os.path.isfile(word2vec_path):
        raise IOError("✘ The word2vec file doesn't exist. "
                      "Please use function <create_vocab_size(embedding_size)"
                      "> to create it!")

    model = word2vec.KeyedVectors.load_word2vec_format(word2vec_path,
                                                       binary=True,
                                                       limit=400000)

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    vector = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            vector[value] = model[key]
    return vector


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果
    shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，
    一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs. Assign 1 for in the test time.
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label
    which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels
