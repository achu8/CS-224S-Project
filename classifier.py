import numpy as np
from convolution import ConvolutionNeuralNetwork
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor, verify_accuracy
from neural_network import LstmNeuralNetwork

class Config:
    random_seed = 101


class Classifier(object):
    def __init__(self, algo, normalize=False, svm_multi_class=None, svm_dual=True, svm_C=1.0, \
                    lstm_vector=None, hidden_layer_sizes=None, learning_rate_init=None, alpha=None, \
                    lstm_state_size=None, lstm_train_dir=None, fc_units=None):
        self.algo = algo
        self.normalize = normalize

        if self.algo == "cnn":
            self.classifier = ConvolutionNeuralNetwork(fc_units=fc_units)

        elif self.algo == "svm":
            if svm_multi_class is None:
                self.classifier = LinearSVC(dual=svm_dual, C=svm_C, random_state=Config.random_seed)
            else:
                self.classifier = LinearSVC(multi_class=svm_multi_class, dual=svm_dual, C=svm_C, random_state=Config.random_seed)
        
        elif self.algo == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        
        elif self.algo == "nn":
            if hidden_layer_sizes is not None:
                self.hidden_layers = hidden_layer_sizes
            else:
                self.hidden_layers = (100,) #(256, 256, 256, 256, 256, 100) # default (100,) # 256, 256, 256 => 
            self.max_iter = 1000 # default 200
            
            if learning_rate_init is not None:
                self.init_lr = learning_rate_init
            else:
                self.init_lr = 0.0001 # default 0.001
                
            self.validation = 0 # default False, no early stopping
            if alpha is not None:
                self.alpha = alpha
            else:
                self.alpha = 0.005
                
            self.classifier = MLPClassifier(solver="adam", \
                                    random_state=Config.random_seed, \
                                    hidden_layer_sizes=self.hidden_layers, \
                                    max_iter=self.max_iter, \
                                    learning_rate_init=self.init_lr, \
                                    early_stopping=self.validation > 0, \
                                    validation_fraction=self.validation, alpha = self.alpha)
            print '**** random seed = ', Config.random_seed
            print '**** NN config ****'
            print 'hidden_layer=', self.hidden_layers, 'max_iter=', self.max_iter, 'learning_rate=', self.init_lr, 'validation=', self.validation, 'alpha=', self.alpha
            print '*******************'
        elif self.algo == "lstm":
            # comment the line below if not running lstm
            self.classifier = LstmNeuralNetwork(vector=lstm_vector, state_size=lstm_state_size, train_dir=lstm_train_dir)
            #raise Exception("Temporarily disabled")

        else:
            raise Exception("unknown mode " + self.algo)
                              
                                    
    def train(self, X, y):
        """ M samples (rows), N features (columns)
        X: each row is a feature vector of an example.
        y: each element is a label.
        X.shape[0] == y.shape[0]
        """
        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std

        self.classifier.fit(X, y)

    def test(self, X, y):
        if self.normalize:
            X = (X - self.mean) / self.std
        
        preds = self.classifier.predict(X)
        accuracy = self.classifier.score(X, y)
        return accuracy, preds

 
# join knn_preds, svm_preds, nn_preds
def combine_preds(preds_list):
    for preds in preds_list:
        assert len(preds) == len(preds_list[0])

    return majority_vote(preds_list)

def majority_vote(preds_list):
    preds_list = [np.reshape(preds, (-1, 1)) for preds in preds_list]
    preds_mat = np.concatenate(preds_list, axis=1)
    n_samples = preds_mat.shape[0]

    y = np.zeros(n_samples, dtype=preds_mat.dtype)
    for i in xrange(n_samples):
        # majority vote
        counts = np.bincount(preds_mat[i]) # cluster's votes
        y[i] = np.argmax(counts) # find the cluster with most votes

    return y

