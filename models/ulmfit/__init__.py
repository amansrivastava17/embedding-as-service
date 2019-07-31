from typing import List, Dict, Optional
import pickle
import tensorflow as tf

from models.ulmfit.model import *

from models import Embedding


class Embeddings(object):
    EMBEDDING_MODELS: List[Embedding] = [
                        Embedding(name=u'umlfit',
                                  dimensions=300,
                                  corpus_size='570k human-generated English sentence pairs',
                                  vocabulary_size='230k',
                                  download_url='http://files.fast.ai/models/wt103/',
                                  format='.h5',
                                  architecture='cbow',
                                  trained_data='Stephen Merity’s Wikitext 103 dataset',
                                  language='en')
                                    ]

    EMBEDDING_MODELS: Dict[str, Embedding] = {embedding.name: embedding for embedding in EMBEDDING_MODELS}

    def __init__(self):
        self.ulmfit_model = None
        self.model = None
        self.word2idx = None
        self.idx2word = None

    @classmethod
    def tokenize(cls, text: str):
        return [word.strip() for word in text.lower().strip().split()]

    def load_model(self, model: str, model_path: str):
        """
            Loads architecture and weights from saved model.
            Args:
                model: Name of the model
                model_path: directory path of saved model and architecture file.

        """
        weights_path = None
        id2word_path = None
        model_files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
        for file in model_files:
            if file.endswith(".h5"):
                weights_path = os.path.join(model_path, file)
            elif file == "itos_wt103.pkl":
                id2word_path = os.path.join(model_path, file)

        with open(id2word_path, 'rb') as f:
            idx2word = pickle.load(f)

        self.word2idx = {word: idx for idx, word in enumerate(idx2word)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.ulmfit_model = build_language_model()
        self.ulmfit_model.load_weights(weights_path)
        self.model = model

    @staticmethod
    def reduce_mean_max(word_embeddings: np.ndarray):
        return tf.concat(values=[tf.reduce_mean(word_embeddings, 0), tf.reduce_max(word_embeddings, 0)], axis=0)

    def encode(self, texts: list, pooling: Optional[str] = None, **kwargs) -> Optional[List[np.array]]:
        pool_func_map = {
            "mean": tf.reduce_mean,
            "max": tf.reduce_max,
            "min": tf.reduce_min,
            "mean_max": Embeddings.reduce_mean_max,
        }
        test_sentences = [Embeddings(test_sentence) for test_sentence in texts]
        X = [[self.word2idx[w] for w in test_sentence] for test_sentence in test_sentences]
        embeddings = []

        for x in X:
            x = np.reshape(x, (1, len(x)))
            embeddings.append(self.ulmfit_model.predict(x)[1][0])
        if not pooling:
            return embeddings
        else:
            if pooling not in ["mean", "max", "mean_max", "min"]:
                print(f"Pooling method \"{pooling}\" not implemented")
                return None
            pooling_func = pool_func_map[pooling]
            pooled = tf.Session().run([pooling_func(word_embeddings, 0) for word_embeddings in embeddings])
            return pooled

    def predict_n_words(self, texts: list, n_words: int = 2) -> List[List[str]]:
        """

        Args:
            texts: List of sentences to predict next words for
            n_words: number of next words to predict for each sentence

        Returns:
            list: list of list of str(n next most likely words for each sentence)
        """

        test_sentences = [Embeddings.tokenize(test_sentence) for test_sentence in texts]
        X = [[self.word2idx[w] for w in test_sentence] for test_sentence in test_sentences]

        results = []
        for x in X:
            results_for_x = []
            for i in range(n_words):
                y = np.reshape(x, (1, len(x)))
                logits = self.ulmfit_model.predict(y)[0]
                answer = np.argmax(logits, axis=2)
                predicted_idx = answer[0][-2]
                x.append(predicted_idx)
                results_for_x.append(self.idx2word[predicted_idx])

            results.append(results_for_x)
        return results
