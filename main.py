import argparse
import model
import os

parser = argparse.ArgumentParser(description='define paramenters for word2vec model')
parser.add_argument('--data_path', type=str, default='./data/corpus.txt', help='path for corpus')
parser.add_argument('--vocab_path', type=str, default='./data/vocab.txt', help='path for corpus')

parser.add_argument('--window_size', type=int, default=3, help='window size')

parser.add_argument('--batch_size', type=int, default=20, help='batch size for each train epoch')
parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimension')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=5000, help='numbers of train epoch')
args = parser.parse_args()

output_path = os.path.join('.', 'save')
if not os.path.exists(output_path):os.makedirs(output_path)
log_path = os.path.join(output_path, 'log.txt')
wordvec_path = os.path.join(output_path, 'wordvecs.txt')


if __name__ == '__main__':

    # prepare data
    sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

    # create vocab(word2id)
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

        for w in context:
            skip_grams.append([target, w])

    model = model.Skipgram(args, vocab_size)
    model.build_model()

    print('====== training ======')
    trained_embeddings = model.train(skip_grams)
    print(trained_embeddings)

