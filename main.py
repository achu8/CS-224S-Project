import numpy as np
import sys, random, collections
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_extractor import FeatureExtractor, verify_accuracy
from sklearn.model_selection import GridSearchCV
from classifier import Classifier, combine_preds
import csv
import os.path

class Config:
    random_seed = 101
    use_test = False
    seq_length = 200
    embed_size = 52#26
    nnlog = 'logNN.csv'
    svmlog = 'logsvm.csv'
    featureset = ['mfcc', 'fbank', 'delta']
    n_classes = 5
    

def main(argv):
    # python main.py lstm nn knn svm all
    # python main.py             # run all
    # python main.py svm all     # only svm
    # python main.py svm pca100
    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    print "argv", argv
    Config.use_test = "test" in argv

    # feature extractor
    extractor = FeatureExtractor()
    
    '''
    ### TODO feature set
    Config.featureset = ["fbank"]
    extractor.__init__(featuresets=["fbank"])
    Print "feature sets used is ", Config.featureset 
    ###
    '''
    
    
    X_train, y_train = extractor.get_train_data()
    X_dev, y_dev = extractor.get_dev_data()
    X_test, y_test = extractor.get_test_data()
    Config.n_classes = extractor.count_lang

    assert y_train.shape == (X_train.shape[0], )
    assert y_dev.shape == (X_dev.shape[0], )
    assert y_test.shape == (X_test.shape[0], )

    lstm_preds_dev, nn_preds_dev = None, None
    lstm_preds_test, nn_preds_test = None, None
    svm_pca80_preds_dev, svm_mfcc_preds_dev, svm_preds_dev = None, None, None
    svm_pca80_preds_test, svm_mfcc_preds_test, svm_preds_test = None, None, None

    if len(argv) == 1 or "cnn" in argv:
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        cnn_X_train = scaler.transform(X_train)
        cnn_X_dev = scaler.transform(X_dev)
        cnn_X_test = scaler.transform(X_test)

        for fc_units in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            cnn_preds_dev, cnn_preds_test = run_classifier("CNN (%d)" % fc_units, "cnn", True, \
                                        cnn_X_train, y_train, cnn_X_dev, y_dev, cnn_X_test, y_test, fc_units=fc_units)


    if len(argv) == 1 or "lstm" in argv:
        # LSTM neural network
        X_mfcc_train, X_fbank_train, X_delta_train, y3_train = extractor.get_train_data(n_inputs=3)
        X_mfcc_dev, X_fbank_dev, X_delta_dev, y3_dev = extractor.get_dev_data(n_inputs=3)
        X_mfcc_test, X_fbank_test, X_delta_test, y3_test = extractor.get_test_data(n_inputs=3)

        assert y3_train.shape == (X_mfcc_train.shape[0], )
        assert y3_train.shape == (X_fbank_train.shape[0], )
        assert y3_train.shape == (X_delta_train.shape[0], )
        
        assert y3_dev.shape == (X_mfcc_dev.shape[0], )
        assert y3_dev.shape == (X_fbank_dev.shape[0], )
        assert y3_dev.shape == (X_delta_dev.shape[0], )

        assert y3_test.shape == (X_mfcc_test.shape[0], )
        assert y3_test.shape == (X_fbank_test.shape[0], )
        assert y3_test.shape == (X_delta_test.shape[0], )

        if "one" in argv and "norm" in argv:
            scaler = StandardScaler()
            scaler.fit(X_fbank_train)
            X_norm_fbank_train = scaler.transform(X_fbank_train)
            X_norm_fbank_dev = scaler.transform(X_fbank_dev)
            X_norm_fbank_test = scaler.transform(X_fbank_test)

            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", True, \
                                X_norm_fbank_train, y3_train, \
                                X_norm_fbank_dev, y3_dev, \
                                X_norm_fbank_test, y3_test, \
                                lstm_vector=(200, 26), lstm_train_dir="data/train.lstm.one")

        if "one" in argv and "norm" not in argv:
            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                X_fbank_train, y3_train, \
                                X_fbank_dev, y3_dev, \
                                X_fbank_test, y3_test, \
                                lstm_vector=(200, 26), lstm_train_dir="data/train.lstm.one")

        if "one100" in argv:
            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                X_fbank_train, y3_train, \
                                X_fbank_dev, y3_dev, \
                                X_fbank_test, y3_test, \
                                lstm_vector=(200, 26), \
                                lstm_state_size=100, lstm_train_dir="data/train.lstm.one.100")

        if "one150" in argv:
            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                X_fbank_train, y3_train, \
                                X_fbank_dev, y3_dev, \
                                X_fbank_test, y3_test, \
                                lstm_vector=(200, 26), \
                                lstm_state_size=150, lstm_train_dir="data/train.lstm.one.150")

        if "one200" in argv:
            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                X_fbank_train, y3_train, \
                                X_fbank_dev, y3_dev, \
                                X_fbank_test, y3_test, \
                                lstm_vector=(200, 26), \
                                lstm_state_size=200, lstm_train_dir="data/train.lstm.one.200")

        if "two" in argv and "norm" in argv:
            scaler = StandardScaler()
            scaler.fit(X_fbank_train)
            X_norm_fbank_train = scaler.transform(X_fbank_train)
            X_norm_fbank_dev = scaler.transform(X_fbank_dev)
            X_norm_fbank_test = scaler.transform(X_fbank_test)

            scaler = StandardScaler()
            scaler.fit(X_mfcc_train)
            X_norm_mfcc_train = scaler.transform(X_mfcc_train)
            X_norm_mfcc_dev = scaler.transform(X_mfcc_dev)
            X_norm_mfcc_test = scaler.transform(X_mfcc_test)

            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                (X_norm_fbank_train, X_norm_mfcc_train), y3_train, \
                                (X_norm_fbank_dev, X_norm_mfcc_dev), y3_dev, \
                                (X_norm_fbank_test, X_norm_mfcc_test), y3_test, \
                                lstm_vector=(200, 26, 13), lstm_train_dir="data/train.lstm.two")


        if "two" in argv and "norm" not in argv:
            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                (X_fbank_train, X_mfcc_train), y3_train, \
                                (X_fbank_dev, X_mfcc_dev), y3_dev, \
                                (X_fbank_test, X_mfcc_test), y3_test, \
                                lstm_vector=(200, 26, 13), lstm_train_dir="data/train.lstm.two")

        if "three" in argv:
            lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                (X_fbank_train, X_mfcc_train, X_delta_train), y3_train, \
                                (X_fbank_dev, X_mfcc_dev, X_delta_dev), y3_dev, \
                                (X_fbank_test, X_mfcc_test, X_delta_test), y3_test, \
                                lstm_vector=(200, 26, 13, 13), lstm_train_dir="data/train.lstm.three")
        #return ########################################################
        
        
        ''' older code to be removed once OK
        lstm_preds_dev, lstm_preds_test = run_classifier("LSTM NN", "lstm", False, \
                                (X_mfcc_train, X_fbank_train, X_delta_train), y3_train, \
                                (X_mfcc_dev, X_fbank_dev, X_delta_dev), y3_dev, \
                                (X_mfcc_test, X_fbank_test, X_delta_test), y3_test, \
                                lstm_vector=(200, 13, 26, 13))
        '''                       
        #return ########################################################

    if len(argv) == 1 or "knn" in argv:
        # k-nearest neighbors
        knn_preds_dev, knn_preds_test = run_classifier("KNN (K=5)", "knn", False, X_train, y_train, X_dev, y_dev, X_test, y_test)

    if len(argv) == 1 or "svm" in argv:
        # convert to PCA featurs
        print "creating PCA"
        print
        if os.path.exists(Config.svmlog):
            with open(Config.svmlog, 'a') as csvfile:
                mywriter = csv.writer(csvfile)
                mywriter.writerow(['================================'])
                mywriter.writerow([argv])
        else:
            with open(Config.svmlog, 'wb') as csvfile:
                mywriter = csv.writer(csvfile)
                mywriter.writerow(['title', 'tr_score', 'dev_score', 'test_score'])
                mywriter.writerow(['================================'])
                mywriter.writerow([argv])
        
        svm_preds_dev, svm_preds_test = run_classifier("SVM (LinearSVC)", "svm", True, \
                            X_train, y_train, X_dev, y_dev, X_test, y_test, svm_C=0.005)
        
        
        if "all" in argv or "pca20" in argv:
            pca20 = PCA(20).fit(X_train)
            X_pca20_train = pca20.transform(X_train)
            X_pca20_dev = pca20.transform(X_dev)
            X_pca20_test = pca20.transform(X_test)
            svm_pca20_preds_dev, svm_pca20_preds_test = run_classifier("SVM (LinearSVC) PCA(20)", "svm", True, \
                            X_pca20_train, y_train, X_pca20_dev, y_dev, X_pca20_test, y_test, svm_dual=False, svm_C=1.0)

        if "all" in argv or "pca50" in argv:
            pca50 = PCA(50).fit(X_train)
            X_pca50_train = pca50.transform(X_train)
            X_pca50_dev = pca50.transform(X_dev)
            X_pca50_test = pca50.transform(X_test)
            svm_pca50_preds_dev, svm_pca50_preds_test = run_classifier("SVM (LinearSVC) PCA(50)", "svm", True, \
                            X_pca50_train, y_train, X_pca50_dev, y_dev, X_pca50_test, y_test, svm_dual=False, svm_C=1.0)

        if "all" in argv or "pca80" in argv:
            pca80 = PCA(80).fit(X_train)
            X_pca80_train = pca80.transform(X_train)
            X_pca80_dev = pca80.transform(X_dev)
            X_pca80_test = pca80.transform(X_test)
            svm_pca80_preds_dev, svm_pca80_preds_test = run_classifier("SVM (LinearSVC) PCA(80)", "svm", True, \
                            X_pca80_train, y_train, X_pca80_dev, y_dev, X_pca80_test, y_test, svm_dual=False, svm_C=1.0)

        if "all" in argv or "pca100" in argv:
            pca100 = PCA(100).fit(X_train)
            X_pca100_train = pca100.transform(X_train)
            X_pca100_dev = pca100.transform(X_dev)
            X_pca100_test = pca100.transform(X_test)
            svm_pca100_preds_dev, svm_pca100_preds_test = run_classifier("SVM (LinearSVC) PCA(100)", "svm", True, \
                            X_pca100_train, y_train, X_pca100_dev, y_dev, X_pca100_test, y_test, svm_dual=False, svm_C=1.0)

        if "all" in argv or "mfccpca" in argv:
            # convert to MFCC-level PCA
            X_mfcc_pca_train, X_mfcc_pca_dev, X_mfcc_pca_test, mfcc_pca_ndims = compute_mfcc_pca(extractor)
            svm_mfcc_preds_dev, svm_mfcc_preds_test = run_classifier("SVM (LinearSVC) MFCC PCA(4)", "svm", True, \
                            X_mfcc_pca_train, y_train, X_mfcc_pca_dev, y_dev, X_mfcc_pca_test, y_test, svm_dual=False, svm_C=0.05)
        
        svm_preds_dev_list = [svm_pca80_preds_dev, svm_mfcc_preds_dev, svm_preds_dev]
        svm_preds_test_list = [svm_pca80_preds_test, svm_mfcc_preds_test, svm_preds_test]
   
    if len(argv) == 1 or "nn" in argv or "nnTrain" in argv:
        # feedforward neural network
        ################################
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)
        ################################
        
        
        if os.path.exists(Config.nnlog):
            with open(Config.nnlog, 'a') as csvfile:
                mywriter = csv.writer(csvfile)
                mywriter.writerow(['================================'])
                mywriter.writerow([argv])
        else:
            with open(Config.nnlog, 'wb') as csvfile:
                print 'here'
                mywriter = csv.writer(csvfile)
                mywriter.writerow(['l_r_i', 'layers', 'alpha', 'tr_score', 'dev_score', 'test_score'])
                mywriter.writerow(['================================'])
                mywriter.writerow([argv])
        
        
        nn_preds_dev, nn_preds_test = run_classifier("NN (MLP)", "nn", False, X_train, y_train, X_dev, y_dev, X_test, y_test)
                

   
    if "nnTrain" in argv: 
        for learning_rate_init in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]: 
            #[0.01, 0.005, 0.001, 0.0005, 0.0001]
            for hidden_layer_sizes in [(100,), (100, 100), (100, 100, 100), (256,), (256, 256), (256, 256, 256)]:  
            # try later  (100,), (100, 100, 100), (256, 256), (256, 256, 256), (256, 256, 256, 100)
                for alpha in [1, 0.1, 0.05, 0.01, 0.005, 0.001, 0]:
                    # [1, 0.1, 0.05, 0.01, 0.005, 0.001, 0]
                    print 'training with learning_rate_init', learning_rate_init, "hidden layer", hidden_layer_sizes, "alpha", alpha
                    nn_preds_dev, nn_preds_test = run_classifier("NN (MLP)", "nn", False, X_train, y_train, X_dev, y_dev, X_test, y_test, hidden_layer_sizes=hidden_layer_sizes,\
                        learning_rate_init=learning_rate_init, alpha=alpha)
    
    
    if len(argv) == 1 or "combine" in argv:
        # combine
        nn_preds_dev_list = [lstm_preds_dev, nn_preds_dev]
        nn_preds_test_list = [lstm_preds_test, nn_preds_test]

        # remove None
        nn_preds_dev_list = [elem for elem in nn_preds_dev_list if elem is not None]
        nn_preds_test_list = [elem for elem in nn_preds_test_list if elem is not None]

        print
        print "**** combine ****"
        preds_dev = combine_preds(svm_preds_dev_list + nn_preds_dev_list + [knn_preds_dev])
        verify_accuracy(preds_dev, y_dev)
        preds_test = combine_preds(svm_preds_test_list + nn_preds_test_list + [knn_preds_test])
        verify_accuracy(preds_test, y_test)
        print


def run_classifier(title, name, normalize, X_train, y_train, X_dev, y_dev, X_test, y_test, \
                    svm_multi_class=None, svm_dual=True, svm_C=1.0, lstm_vector=None,\
                    hidden_layer_sizes=None, learning_rate_init=None, alpha=None, \
                    lstm_state_size=None, lstm_train_dir=None, fc_units=None):
  
    print "**** %s ****" % title
      
    classifier = Classifier(name, normalize=normalize, \
        svm_multi_class=svm_multi_class, svm_dual=svm_dual, svm_C=svm_C, \
        lstm_vector=lstm_vector, hidden_layer_sizes=hidden_layer_sizes, \
        learning_rate_init=learning_rate_init, alpha=alpha, lstm_train_dir=lstm_train_dir, \
        fc_units=fc_units)

    
    classifier.train(X_train, y_train) 
    accuracy_train, _ = classifier.test(X_train, y_train)
    
  
         
    print "Train Accuracy = %6.4f (data %snormalized)" % (accuracy_train, "" if normalize else "not ")
    
    accuracy_dev, preds_dev = classifier.test(X_dev, y_dev)
    print "Dev Accuracy = %6.4f (data %snormalized)" % (accuracy_dev, "" if normalize else "not ")
    verify_accuracy(preds_dev, y_dev)
    print "Confusion Matrix"
    print compute_confusion_matrix(Config.n_classes, y_dev, preds_dev)
    
    accuracy_test, preds_test = 0, []
    if Config.use_test:
        accuracy_test, preds_test = classifier.test(X_test, y_test)
        print "****Test Accuracy**** = %6.4f (data %snormalized)" % (accuracy_test, "" if normalize else "not ")
        verify_accuracy(preds_test, y_test)
        print "Confusion Matrix"
        print compute_confusion_matrix(Config.n_classes, y_test, preds_test)
        print "------------"
    
    if title == "NN (MLP)" :
         print 'writing csv file', classifier.init_lr, classifier.hidden_layers, classifier.alpha, accuracy_train, accuracy_dev, accuracy_test
         lr = classifier.init_lr
         layer = classifier.hidden_layers
         alpha1 = classifier.alpha
         with open(Config.nnlog, 'a') as f:
            mywriter = csv.writer(f)
            mywriter.writerow([lr, layer, alpha1, accuracy_train, accuracy_dev, accuracy_test])
            
            
    if name == "svm" :
        with open(Config.svmlog, 'a') as csvfile:
            print title, accuracy_train, accuracy_dev, accuracy_test
            mywriter = csv.writer(csvfile)
            mywriter.writerow([title, accuracy_train, accuracy_dev, accuracy_test])
            
            
    # run a grid search on a SVM
    
    svm_Cs = np.array([1,0.1]) #adding more
    if title == "SVM (LinearSVC) PCA(80)" or title == "SVM (LinearSVC) PCA(50)" \
            or title == "SVM (LinearSVC) PCA(20)" or title == "SVM (LinearSVC) PCA(100)" \
            or title == "SVM (LinearSVC) MFCC PCA(4)" :
        print "---------------"
        print "*** grid search on ***", title
        # create and fit a ridge regression model, testing each C
        model = classifier.classifier
        grid = GridSearchCV(estimator=model, param_grid=dict(C=svm_Cs))
        #grid.fit(X_train, y_train)
        grid.fit(np.concatenate((X_train, X_dev), axis=0), np.concatenate((y_train, y_dev), axis=0))
        print(grid)
        # summarize the results of the grid search
        print(grid.best_score_)
        print(grid.best_estimator_.C)
        print "---------------"
       
     # run a grid search on a NN
    
    ''' DON'T USE THIS
    if title == "NN (MLP)"  :
        alpha = np.array ([1, 0.1])#([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001])
        hidden_layer_sizes = np.array([(100), (256)])
        #hidden_layer_sizes = np.array([(100), (100, 100), (100, 100, 100), (256), (256, 256), (256, 256, 256), (256, 256, 256, 100)])
        learning_rate_init = np.array([0.001]) #([0.01, 0.005, 0.001, 0.0005, 0.0001])
        
        print "---------------"
        print "*** grid search on ***", title
        # create and fit a ridge regression model, testing each 
        model = classifier.classifier
        grid = GridSearchCV(estimator=model, param_grid=dict(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init = learning_rate_init, alpha = alpha))
        grid.fit(X_train, y_train)
        print(grid)
        # summarize the results of the grid search
        print(grid.best_score_)
        print(grid.best_estimator_.alpha)
        print(grid.best_estimator_.hidden_layer_sizes)
        print(grid.best_estimator_.learning_rate_init)
        print "---------------"
    '''
    return preds_dev, preds_test

def compute_mfcc_pca(extractor): # apply PCA to individual MFCC
    X_raw, _ = extractor.get_train_data(flatten=False)
    n_samples, seq_length, embed_size = X_raw.shape
    mfcc_pca_ndims = int(embed_size / 3)
    #print "mfcc_pca_ndims", mfcc_pca_ndims
    
    pca_list = [PCA(mfcc_pca_ndims).fit(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i in xrange(seq_length)]
    X_mfcc_pca_train = np.concatenate([pca.transform(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i, pca in enumerate(pca_list)], axis=1)
    assert X_mfcc_pca_train.shape[1] == seq_length * mfcc_pca_ndims

    X_raw, _ = extractor.get_dev_data(flatten=False)
    X_mfcc_pca_dev = np.concatenate([pca.transform(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i, pca in enumerate(pca_list)], axis=1)
    assert X_mfcc_pca_dev.shape[1] == seq_length * mfcc_pca_ndims

    X_raw, _ = extractor.get_test_data(flatten=False)
    X_mfcc_pca_test = np.concatenate([pca.transform(np.reshape(X_raw[:, i, :], (-1, embed_size))) for i, pca in enumerate(pca_list)], axis=1)
    assert X_mfcc_pca_test.shape[1] == seq_length * mfcc_pca_ndims

    return X_mfcc_pca_train, X_mfcc_pca_dev, X_mfcc_pca_test, mfcc_pca_ndims

def get_embed_size():
    return Config.embed_size

def compute_confusion_matrix(n_labels, labels, predictions):
    n_samples = len(predictions)
    assert n_samples == len(labels)
    
    match_count = collections.defaultdict(lambda: int())
    mismatch_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: int()))
    
    for i, lab in enumerate(labels):
        pred = predictions[i]
        if lab == pred:
            match_count[lab] += 1
        else:
            mismatch_dict[lab][pred] += 1

    # row label: label
    # column label: prediction
    c_mat = np.zeros((n_labels, n_labels), dtype=np.int)
    for lab in xrange(n_labels):
        c_mat[lab, lab] = match_count[lab]
        for pred in xrange(n_labels):
            if lab != pred:
                c_mat[lab, pred] = mismatch_dict[lab][pred]

    return c_mat

if __name__ == "__main__":
    main(sys.argv)
