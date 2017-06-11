import numpy as np
from sklearn.model_selection import train_test_split
from numpy import random,argsort,sqrt
from pylab import plot,show
import pickle
import sys, random

class Config:
    min_seq_length = 200 #50 #200

class Label:
    tamil = 0 #TA
    germany = 1  #GE
    brazilianP = 2 #BP
    hindi = 3 #HI
    spanish = 4 #SP
    arabic = 10
    cantonese = 11
    french = 12
    japanese = 13

class Show:
    tamil = False #TA
    germany = False  #GE
    brazilianP = False #BP
    hindi = False #HI
    spanish = False #SP
    arabic = False
    cantonese = False
    french = False
    japanese = False

    
class FeatureExtractor(object):
    def __init__(self, dev_split=0.2, test_split=0.2, n_samples_per_lang=30000, featuresets=["mfcc", "fbank", "delta"]):
        self.dev_split = dev_split
        self.test_split = test_split
        self.n_samples_per_lang = n_samples_per_lang
        self.seq_length, self.embed_size = None, None # (50, 13)
        self.featuresets = featuresets
        
        print "Loading data"
        
        print 'loading mfcc'
        #TODO adding/dropping languages
        ta = pickle.load(open('data/mfcc_features/TA', 'rb'))
        ge = pickle.load(open('data/mfcc_features/GE', 'rb'))
        bp = pickle.load(open('data/mfcc_features/BP', 'rb'))
        hi = pickle.load(open('data/mfcc_features/HI', 'rb'))
        sp = pickle.load(open('data/mfcc_features/SP', 'rb'))
        #ca = pickle.load(open('data/mfcc_features/CA', 'rb'))
        
        print 'loading delta'
        #TODO adding/dropping languages
        ta_delta = pickle.load(open('data/delta_features/TA', 'rb'))
        ge_delta = pickle.load(open('data/delta_features/GE', 'rb'))
        bp_delta = pickle.load(open('data/delta_features/BP', 'rb'))
        hi_delta = pickle.load(open('data/delta_features/HI', 'rb'))
        sp_delta = pickle.load(open('data/delta_features/SP', 'rb'))
        #ca_delta = pickle.load(open('data/delta_features/CA', 'rb'))
        
        print 'loading fbank'
        #TODO adding/dropping languages
        ta_fbank = pickle.load(open('data/fbank_features/TA', 'rb'))
        ge_fbank = pickle.load(open('data/fbank_features/GE', 'rb'))
        bp_fbank = pickle.load(open('data/fbank_features/BP', 'rb'))
        hi_fbank = pickle.load(open('data/fbank_features/HI', 'rb'))
        sp_fbank = pickle.load(open('data/fbank_features/SP', 'rb'))
        #ca_fbank = pickle.load(open('data/fbank_features/CA', 'rb'))

        print "Filling the data set"
        self.X_all, self.y_all = [], []
        self.X_train, self.y_train = [], []
        self.X_dev, self.y_dev = [], []
        self.X_test, self.y_test = [], []

        self.X3_all, self.y3_all = [], []
        self.y3_train, self.y3_dev, self.y3_test = [], [], []
        self.X_mfcc_train, self.X_mfcc_dev, self.X_mfcc_test = [], [], []
        self.X_delta_train, self.X_delta_dev, self.X_delta_test = [], [], []
        self.X_fbank_train, self.X_fbank_dev, self.X_fbank_test = [], [], []

        if "mfcc" in featuresets and "fbank" in featuresets and "delta" in featuresets :
            print "*** mfcc delta fbank all featuresets"
        elif "mfcc" in featuresets and "fbank" in featuresets :
            print "*** mfcc fbank featuresets"
        elif "mfcc" in featuresets and "delta" in featuresets :
            print "*** mfcc delta featuresets"
        elif "delta" in featuresets and "fbank" in featuresets :
            print "*** delta fbank featuresets"
        elif "mfcc" in featuresets  :
            print "*** mfcc featureset only"
        elif "delta" in featuresets  :
            print "*** delta featureset only"
        elif "fbank" in featuresets :
            print "*** fbank featureset only"
        else:
            print "*** mfcc delta fbank all featuresets"
        
        
        self.fill100 = False ####TODO 100 samples
        if not self.fill100:
            print '*** all samples'
            ### use all samples available and then split 80/20
            '''
            self.OLD_fill(ar, 'AR')
            self.OLD_fill(ca, 'CA')
            self.OLD_fill(fr, 'FR')
            self.OLD_fill(ja, 'JA')
            self.OLD_fill(sp, 'SP')
            '''
           
            #TODO adding/dropping languages
            self.fill3(ta, ta_delta, ta_fbank, 'TA')
            self.fill3(ge, ge_delta, ge_fbank, 'GE')
            self.fill3(bp, bp_delta, bp_fbank, 'BP')
            self.fill3(hi, hi_delta, hi_fbank, 'HI')
            self.fill3(sp, sp_delta, sp_fbank, 'SP')
            #self.fill3(ca, ca_delta, ca_fbank, 'CA')
            
            
            #TODO adding/dropping languages
            self.fillType(ta, ta_delta, ta_fbank, 'TA')
            self.fillType(ge, ge_delta, ge_fbank, 'GE')
            self.fillType(bp, bp_delta, bp_fbank, 'BP')
            self.fillType(hi, hi_delta, hi_fbank, 'HI')
            self.fillType(sp, sp_delta, sp_fbank, 'SP')
            #self.fillType(ca, ca_delta, ca_fbank, 'CA')
            ###
            
        else:
            print '***100 samples only'
            #### use only 100 samples and then split 80/20
            '''
            self.OLD_fillOnly100samples(ar, 'AR')
            self.OLD_fillOnly100samples(ca, 'CA')
            self.OLD_fillOnly100samples(fr, 'FR')
            self.OLD_fillOnly100samples(ja, 'JA')
            self.OLD_fillOnly100samples(sp, 'SP')
            '''
            
            #TODO adding/dropping languages
            #self.OLD_fillOnly100samples(ta, 'TA')
            self.OLD_fillOnly100samples(ge, 'GE')
            #self.OLD_fillOnly100samples(bp, 'BP')
            self.OLD_fillOnly100samples(hi, 'HI')
            #self.OLD_fillOnly100samples(sp, 'SP')
            self.OLD_fillOnly100samples(ca, 'CA')
            
        # convert data to meet sklearn interface
        self.X_all = np.array(self.X_all)
        self.y_all = np.array(self.y_all)
        print "X_all.shape", self.X_all.shape, "y_all.shape", self.y_all.shape


        self.X_train = np.concatenate(self.X_train, axis=0)
        self.y_train = np.concatenate(self.y_train, axis=0)
        self.X_train, self.y_train = shuffle_data(self.X_train, self.y_train)
        print "X_train.shape", self.X_train.shape, "y_train.shape", self.y_train.shape
        
        self.X_dev = np.concatenate(self.X_dev, axis=0)
        self.y_dev = np.concatenate(self.y_dev, axis=0)
        self.X_dev, self.y_dev = shuffle_data(self.X_dev, self.y_dev)
        print "X_dev.shape", self.X_dev.shape, "y_dev.shape", self.y_dev.shape

        self.X_test = np.concatenate(self.X_test, axis=0)
        self.y_test = np.concatenate(self.y_test, axis=0)
        self.X_test, self.y_test = shuffle_data(self.X_test, self.y_test)
        print "X_test.shape", self.X_test.shape, "y_test.shape", self.y_test.shape

        # train for mfcc, delta, fbank
        mfcc = np.concatenate(self.X_mfcc_train, axis=0)
        delta = np.concatenate(self.X_delta_train, axis=0)
        fbank = np.concatenate(self.X_fbank_train, axis=0)
        y3 = np.concatenate(self.y3_train, axis=0)
        self.X_mfcc_train, self.y3_train, self.X_delta_train, self.X_fbank_train = shuffle_data(mfcc, y3, X2=delta, X3=fbank)
        
        print "X_mfcc_train.shape", self.X_mfcc_train.shape
        print "X_delta_train.shape", self.X_delta_train.shape
        print "X_fbank_train.shape", self.X_fbank_train.shape
        print "y3_train.shape", self.y3_train.shape

        # dev for mfcc, delta, fbank
        mfcc = np.concatenate(self.X_mfcc_dev, axis=0)
        delta = np.concatenate(self.X_delta_dev, axis=0)
        fbank = np.concatenate(self.X_fbank_dev, axis=0)
        y3 = np.concatenate(self.y3_dev, axis=0)
        self.X_mfcc_dev, self.y3_dev, self.X_delta_dev, self.X_fbank_dev = shuffle_data(mfcc, y3, X2=delta, X3=fbank)
        
        print "X_mfcc_dev.shape", self.X_mfcc_dev.shape
        print "X_delta_dev.shape", self.X_delta_dev.shape
        print "X_fbank_dev.shape", self.X_fbank_dev.shape
        print "y3_dev.shape", self.y3_dev.shape

        # test for mfcc, delta, fbank
        mfcc = np.concatenate(self.X_mfcc_test, axis=0)
        delta = np.concatenate(self.X_delta_test, axis=0)
        fbank = np.concatenate(self.X_fbank_test, axis=0)
        y3 = np.concatenate(self.y3_test, axis=0)
        self.X_mfcc_test, self.y3_test, self.X_delta_test, self.X_fbank_test = shuffle_data(mfcc, y3, X2=delta, X3=fbank)

        print "X_mfcc_test.shape", self.X_mfcc_test.shape
        print "X_delta_test.shape", self.X_delta_test.shape
        print "X_fbank_test.shape", self.X_fbank_test.shape
        print "y3_test.shape", self.y3_test.shape

        print 
        print "All data breakdown", len(self.X_all)
        
        self.count_lang = 0
        if np.sum(self.y_all == Label.arabic) > 0:
            Show.arabic = True
            print "    arabic =", np.sum(self.y_all == Label.arabic)
            self.count_lang += 1
            
        if np.sum(self.y_all == Label.cantonese)> 0:    
            print "    cantonese =", np.sum(self.y_all == Label.cantonese)
            Show.cantonese = True
            self.count_lang += 1
        
        if np.sum(self.y_all == Label.french) > 0 :
            print "    french =", np.sum(self.y_all == Label.french)
            Show.french = True
            self.count_lang += 1
            
        if np.sum(self.y_all == Label.japanese) > 0 :    
            print "    japanese =", np.sum(self.y_all == Label.japanese)
            Show.japanese = True  
            self.count_lang += 1   
            
        if np.sum(self.y_all == Label.tamil) > 0 :
            print "    Tamil =", np.sum(self.y_all == Label.tamil)
            Show.tamil = True
            self.count_lang += 1
            
        if np.sum(self.y_all == Label.germany) > 0 :
            print "    Germany =", np.sum(self.y_all == Label.germany)
            Show.germany = True
            self.count_lang += 1
            
        if np.sum(self.y_all == Label.brazilianP) > 0:
            print "    Brazilian P =", np.sum(self.y_all == Label.brazilianP)
            Show.brazilianP = True
            self.count_lang += 1
            
        if np.sum(self.y_all == Label.hindi) > 0:
            print "    Hindi =", np.sum(self.y_all == Label.hindi)
            Show.hindi = True
            self.count_lang += 1
            
        if np.sum(self.y_all == Label.spanish) > 0:
            print "    spanish =", np.sum(self.y_all == Label.spanish)
            Show.spanish = True
            self.count_lang += 1
        
        print "****", self.count_lang, "languages classification", "****"
        
    def get_train_data(self, n_inputs=1, flatten=True):
        if n_inputs == 2:
            X_mfcc_train = self.X_mfcc_train
            X_fbank_train = self.X_fbank_train
            if flatten:
                X_mfcc_train = np.reshape(X_mfcc_train, (X_mfcc_train.shape[0], -1))
                X_fbank_train = np.reshape(X_fbank_train, (X_fbank_train.shape[0], -1))
            return X_mfcc_train, X_fbank_train, self.y3_train

        if n_inputs == 3:
            X_mfcc_train = self.X_mfcc_train
            X_fbank_train = self.X_fbank_train
            X_delta_train = self.X_delta_train
            if flatten:
                X_mfcc_train = np.reshape(X_mfcc_train, (X_mfcc_train.shape[0], -1))
                X_fbank_train = np.reshape(X_fbank_train, (X_fbank_train.shape[0], -1))
                X_delta_train = np.reshape(X_delta_train, (X_delta_train.shape[0], -1))
            return X_mfcc_train, X_fbank_train, X_delta_train, self.y3_train

        # default, n_inputs == 1
        X_train = self.X_train
        if flatten:
            X_train = np.reshape(X_train, (X_train.shape[0], -1))
        return X_train, self.y_train


    def get_dev_data(self, n_inputs=1, flatten=True):
        if n_inputs == 2:
            X_mfcc_dev = self.X_mfcc_dev
            X_fbank_dev = self.X_fbank_dev
            if flatten:
                X_mfcc_dev = np.reshape(X_mfcc_dev, (X_mfcc_dev.shape[0], -1))
                X_fbank_dev = np.reshape(X_fbank_dev, (X_fbank_dev.shape[0], -1))
            return X_mfcc_dev, X_fbank_dev, self.y3_dev

        if n_inputs == 3:
            X_mfcc_dev = self.X_mfcc_dev
            X_fbank_dev = self.X_fbank_dev
            X_delta_dev = self.X_delta_dev
            if flatten:
                X_mfcc_dev = np.reshape(X_mfcc_dev, (X_mfcc_dev.shape[0], -1))
                X_fbank_dev = np.reshape(X_fbank_dev, (X_fbank_dev.shape[0], -1))
                X_delta_dev = np.reshape(X_delta_dev, (X_delta_dev.shape[0], -1))
            return X_mfcc_dev, X_fbank_dev, X_delta_dev, self.y3_dev

        # default, n_inputs == 1
        X_dev = self.X_dev
        if flatten:
            X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
        return X_dev, self.y_dev


    def get_test_data(self, n_inputs=1, flatten=True):
        if n_inputs == 2:
            X_mfcc_test = self.X_mfcc_test
            X_fbank_test = self.X_fbank_test
            if flatten:
                X_mfcc_test = np.reshape(X_mfcc_test, (X_mfcc_test.shape[0], -1))
                X_fbank_test = np.reshape(X_fbank_test, (X_fbank_test.shape[0], -1))
            return X_mfcc_test, X_fbank_test, self.y3_test

        if n_inputs == 3:
            X_mfcc_test = self.X_mfcc_test
            X_fbank_test = self.X_fbank_test
            X_delta_test = self.X_delta_test
            if flatten:
                X_mfcc_test = np.reshape(X_mfcc_test, (X_mfcc_test.shape[0], -1))
                X_fbank_test = np.reshape(X_fbank_test, (X_fbank_test.shape[0], -1))
                X_delta_test = np.reshape(X_delta_test, (X_delta_test.shape[0], -1))
            return X_mfcc_test, X_fbank_test, X_delta_test, self.y3_test
            
        # default, n_inputs == 1
        X_test = self.X_test
        if flatten:
            X_test = np.reshape(X_test, (X_test.shape[0], -1))
        return X_test, self.y_test


    def fill3(self, data, delta, fbank, label): # data stores mfcc features
        X_batch, y_batch = [], []
        n_samples = 0
        for sample, mfcc in data.items():
            if sample == 260 and label == 'CA': break
            if n_samples >= self.n_samples_per_lang: break

            # ignore bad data
            if mfcc.shape[0] < Config.min_seq_length:
                print "bad mfcc: y =", label, "sample=", sample, "shape =", mfcc.shape
                continue
            
            if delta[sample].shape[0] < Config.min_seq_length:
                print "bad delta: y =", label, "sample=", sample, "shape =", delta[sample].shape
                continue
            
            if fbank[sample].shape[0] < Config.min_seq_length:
                print "bad fbank: y =", label, "sample=", sample, "shape =", fbank[sample].shape
                continue

            if "mfcc" in self.featuresets and "fbank" in self.featuresets and "delta" in self.featuresets :
                # print "mfcc delta fbank all featuresets"
                features = np.concatenate((data[sample], delta[sample], fbank[sample]), axis=1)
            elif "mfcc" in self.featuresets and "fbank" in self.featuresets :
                features = np.concatenate((data[sample], fbank[sample]), axis=1)
            elif "mfcc" in self.featuresets and "delta" in self.featuresets :
                features = np.concatenate((data[sample], fbank[sample]), axis=1)
            elif "delta" in self.featuresets and "fbank" in self.featuresets :
                features = np.concatenate((fbank[sample], delta[sample]), axis=1)
            elif "mfcc" in self.featuresets  :
                features = data[sample]
            elif "delta" in self.featuresets  :
                features = delta[sample]
            elif "fbank" in self.featuresets :
                features = fbank[sample]
            else:
                features = np.concatenate((data[sample], delta[sample], fbank[sample]), axis=1)
                
                
            if label == "TA": y = Label.tamil
            elif label == "GE": y = Label.germany
            elif label == "BP": y = Label.brazilianP
            elif label == "HI": y = Label.hindi
            elif label == "SP": y = Label.spanish
            elif label == "AR": y = Label.arabic
            elif label == "CA": y = Label.cantonese
            elif label == "FR": y = Label.french
            elif label == "JA": y = Label.japanese
            else: continue

            n_samples += 1
            
            if self.seq_length is None:
                self.seq_length, self.embed_size = features.shape
                print "seq_length", self.seq_length, "embed_size", self.embed_size
        
            self.X_all.append(features)
            self.y_all.append(y)
            
            X_batch.append(features)
            y_batch.append(y)

        
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        X_batch, y_batch = shuffle_data(X_batch, y_batch)
        
        test_limit = int(self.test_split * len(data))
        dev_limit = int(self.dev_split * len(data))
        train_limit = len(data) - dev_limit - test_limit
        
        self.X_train.append(X_batch[ : train_limit])
        self.y_train.append(y_batch[ : train_limit])

        self.X_dev.append(X_batch[train_limit : dev_limit+train_limit])
        self.y_dev.append(y_batch[train_limit : dev_limit+train_limit])

        self.X_test.append(X_batch[dev_limit+train_limit : ])
        self.y_test.append(y_batch[dev_limit+train_limit : ])


    def fillType(self, data, delta, fbank, label): # data stores mfcc features
        X_mfcc_batch, X_delta_batch, X_fbank_batch, y3_batch = [], [], [], []
        n_samples = 0
        for sample, mfcc in data.items():
            if sample == 260 and label == 'CA': break
            if n_samples >= self.n_samples_per_lang: break

            # ignore bad data
            if mfcc.shape[0] < Config.min_seq_length:
                print "bad mfcc: y =", label, "sample=", sample, "shape =", mfcc.shape
                continue
            
            if delta[sample].shape[0] < Config.min_seq_length:
                print "bad delta: y =", label, "sample=", sample, "shape =", delta[sample].shape
                continue
            
            if fbank[sample].shape[0] < Config.min_seq_length:
                print "bad fbank: y =", label, "sample=", sample, "shape =", fbank[sample].shape
                continue

   
            if label == "TA": y = Label.tamil
            elif label == "GE": y = Label.germany
            elif label == "BP": y = Label.brazilianP
            elif label == "HI": y = Label.hindi
            elif label == "SP": y = Label.spanish
            elif label == "AR": y = Label.arabic
            elif label == "CA": y = Label.cantonese
            elif label == "FR": y = Label.french
            elif label == "JA": y = Label.japanese
            else: continue

            n_samples += 1

            self.X3_all.append((data[sample], delta[sample], fbank[sample]))
            self.y3_all.append(y)
            
            X_mfcc_batch.append(data[sample])
            X_delta_batch.append(delta[sample])
            X_fbank_batch.append(fbank[sample])          
            y3_batch.append(y)


        X_mfcc_batch = np.array(X_mfcc_batch)
        X_delta_batch = np.array(X_delta_batch)
        X_fbank_batch = np.array(X_fbank_batch)
        y3_batch = np.array(y3_batch)

        X_mfcc_batch, y_batch, X_delta_batch, X_fbank_batch = shuffle_data(X_mfcc_batch, y3_batch, X2=X_delta_batch, X3=X_fbank_batch)
        
        test_limit = int(self.test_split * len(data))
        dev_limit = int(self.dev_split * len(data))
        train_limit = len(data) - dev_limit - test_limit
        
        self.X_mfcc_train.append(X_mfcc_batch[ : train_limit])
        self.X_mfcc_dev.append(X_mfcc_batch[train_limit : dev_limit+train_limit])
        self.X_mfcc_test.append(X_mfcc_batch[dev_limit+train_limit : ])

        self.X_delta_train.append(X_delta_batch[ : train_limit])
        self.X_delta_dev.append(X_delta_batch[train_limit : dev_limit+train_limit])
        self.X_delta_test.append(X_delta_batch[dev_limit+train_limit : ])

        self.X_fbank_train.append(X_fbank_batch[ : train_limit])
        self.X_fbank_dev.append(X_fbank_batch[train_limit : dev_limit+train_limit])
        self.X_fbank_test.append(X_fbank_batch[dev_limit+train_limit : ])

        self.y3_train.append(y3_batch[ : train_limit])
        self.y3_dev.append(y3_batch[train_limit : dev_limit+train_limit])
        self.y3_test.append(y3_batch[dev_limit+train_limit : ])

         
            
    def OLD_fillOnly100samples(self, data, label): ### only fill with the first 100 samples for each language
        X_train_batch, y_train_batch = [], []
        X_test_batch, y_test_batch = [], []
        i = 0      
        for sample, features in data.items():
            if i == 100: break
            if i < 80: # training set 
                ## ignore bad data
                #print 'here', i
                if features.shape[0] < Config.min_seq_length:
                   print "bad: y =", label, "shape =", features.shape
                   continue
                '''
                if label == "AR": y = Label.arabic
                elif label == "CA": y = Label.cantonese
                elif label == "FR": y = Label.french
                elif label == "JA": y = Label.japanese
                elif label == "SP": y = Label.spanish
                '''
                if label == "TA": y = Label.tamil
                elif label == "GE": y = Label.germany
                elif label == "BP": y = Label.brazilianP
                elif label == "HI": y = Label.hindi
                elif label == "SP": y = Label.spanish
                else: continue
                #print 'there'

                X_train_batch.append(features)
                y_train_batch.append(y)

            else: ### test set
                
                X_test_batch.append(features)
                y_test_batch.append(y)
                
            self.X_all.append(features)
            self.y_all.append(y)
                
            if self.seq_length is None:
                    self.seq_length, self.embed_size = features.shape
                    print "seq_length", self.seq_length, "embed_size", self.embed_size        
            i+= 1

        X_train_batch = np.array(X_train_batch)
        y_train_batch = np.array(y_train_batch)
        X_train_batch, y_train_batch = shuffle_data(X_train_batch, y_train_batch)

        X_test_batch = np.array(X_test_batch)
        y_test_batch = np.array(y_test_batch)
        X_test_batch, y_test_batch = shuffle_data(X_test_batch, y_test_batch)

        self.X_train.append(X_train_batch)
        self.y_train.append(y_train_batch)
        self.X_test.append(X_test_batch)
        self.y_test.append(y_test_batch)
           

no_shuffle = False
def shuffle_data(X, y, X2=None, X3=None):
    if no_shuffle:
        return X, y

    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    
    if X2 is not None and X3 is None:
        return X[indices], y[indices], X2[indices]
    
    if X3 is not None:
        assert X2 is not None
        return X[indices], y[indices], X2[indices], X3[indices]
    
    return X[indices], y[indices]


def verify_accuracy(labels, actual):
    assert len(labels) == len(actual)
    a, c, f, j, s, t, g, b, h = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    total_a = float(np.sum(actual == Label.arabic))
    total_c = float(np.sum(actual == Label.cantonese))
    total_f = float(np.sum(actual == Label.french))
    total_j = float(np.sum(actual == Label.japanese))
    total_s = float(np.sum(actual == Label.spanish))
   
    total_t = float(np.sum(actual == Label.tamil))
    total_g = float(np.sum(actual == Label.germany))
    total_b = float(np.sum(actual == Label.brazilianP))
    total_h = float(np.sum(actual == Label.hindi))
    
    total = total_a + total_c + total_f + total_j + total_s + total_t + total_g + total_b + total_h
    assert total == float(len(actual))
    
    for i in xrange(len(labels)):
        if labels[i] == actual[i]:
            
            if labels[i] == Label.arabic: a += 1
            elif labels[i] == Label.cantonese: c += 1
            elif labels[i] == Label.french: f += 1
            elif labels[i] == Label.japanese: j += 1
            elif labels[i] == Label.spanish: s += 1
            elif labels[i] == Label.tamil: t += 1
            elif labels[i] == Label.germany: g += 1
            elif labels[i] == Label.brazilianP: b += 1
            elif labels[i] == Label.hindi: h += 1
            else:
                raise Exception("unknown label %d" % labels[i])

    if Show.arabic:
        print "Arabic: %6.4f" % (a / safe_denominator(total_a))
    
    if Show.cantonese:
        print "Cantonese: %6.4f" % (c / safe_denominator(total_c))
        
    if Show.french:
        print "French: %6.4f" % (f / safe_denominator(total_f))
        
    if Show.japanese:   
        print "Japanese: %6.4f" % (j / safe_denominator(total_j))
        
    if Show.spanish:
        print "Spanish: %6.4f" % (s / safe_denominator(total_s))
   
    if Show.tamil:
        print "tamil: %6.4f" % (t / safe_denominator(total_t))
    
    if Show.germany:
        print "germany: %6.4f" % (g / safe_denominator(total_g))
        
    if Show.brazilianP:
        print "brazilianP: %6.4f" % (b / safe_denominator(total_b))
        
    if Show.hindi:
        print "Hindi: %6.4f" % (h / safe_denominator(total_h))
    
    print "Total correct: %6.4f" % ((a+c+f+j+s+t+g+b+h) / total)

def safe_denominator(a):
    return a if a != 0.0 else 1.0
 
