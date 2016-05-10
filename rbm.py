from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import preprocessing

import sys
import optparse
import os
import numpy as np
from compiler.ast import flatten

def checkBoard(annotation, boards):
   
    if (boards==[0,1] or boards==[1,0] or boards==[0,0]):
        return annotation == boards 
    elif boards==[1,1] and not annotation==[0,0]:
            return True
    else:
        pass    

def parseFile (fn, fn_board_annotation, boards, features, sample_rate=1):
    all_features = []
    
    
    f = open(fn, 'r')
    f_annotation = open(fn_board_annotation, 'r')

    # fixme take out this different format
    f_annotation.readline()
    
    def cutline(line, divchar, fr, num):
        line = line.rstrip()
        line = line.split(divchar)
        line = map(int, line[fr:fr+num])
        return line
    
    for  (line, annotation) in zip (f, f_annotation)[::sample_rate]:
        annotation = cutline(annotation, ' ', 2, 2)
        if checkBoard (annotation,boards):
            line = line.rstrip()
            line = line.split(',')
            frame = int (line[0])
            featurevec = [line [features[i]]  for i in range(len(features))]
            all_features.append((frame,featurevec))
    f.close()
    f_annotation.close()

    return all_features



#FIXME bug with 2 frames lag in distances

def composeFeatures ():
    def empty(featurevec):
        if ('' in featurevec ) or (' ' in featurevec):
            return True
    def equal_ts(tss): 
        tss= map (int, tss)
        return all(x==tss[0] for x in tss)
    def get_ts(vec):
        return vec[0]
    def get_feature(vec):
        return vec[1]
    def map_labels_spread_to_fingers(labels,maxlabel):
         labeli = map (lambda x,y: x+y, labels, [0,maxlabel,2*maxlabel,3*maxlabel,4*maxlabel])
         return labeli

    #groups: 0(unclear), 1 (on),2 (off), 3 (slip), 4  (luft)
    def map_labels_4 (labels):
        newlabels=[]

        dic = {0:0, 1:1, 2:2, 3:3, 4:3, 5:3, 6:3, 7:3,8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3, 15:4 }
        for i in range(5):
            newlabels.append(dic[labels[i]])
        return newlabels
            
           
        
    alldata=[]
    labeling=[]
    for i  in range(len(distances)):
        di=distances[i]
        vfi=fingers[i]
        vhi=hand[i]
        labeli = labels[i]

        if equal_ts(map(get_ts, [vfi,vhi, labeli])):
            features= map (get_feature, [di, vfi,vhi])
            if not (True in  map (empty, features)):
                features= flatten (features)
                labeli= get_feature(labeli)
                labeli = map( int, labeli)
                
                #FIXME: dirty hack for  multilabel classification 
                labeli= map_labels_4(labeli)
                labeli = map_labels_spread_to_fingers(labeli, 5)
                labeling += [labeli]
            
                alldata +=[map(float, features)]
                
    X = np.array(alldata)
    Y = np.array(labeling) 
    
    Y= MultiLabelBinarizer().fit_transform(Y)
    return X,Y 
                

                
            


# FIXME: assert the similarity of timestamps

if __name__ == "__main__":
    parser = optparse.OptionParser(usage = 'usage: %prog [OPTIONS]')
    parser.add_option('--handv-file', 
                      dest    = 'handv_file',
                      default = 'handvelocity.txt')
    parser.add_option('--fingerv-file', 
                      dest    = 'fingerv_file',
                      default = 'fingervelocity.txt')
    parser.add_option('--distance-file', 
                      dest    = 'distance_file',
                      default = 'smootheddistances.txt')
    parser.add_option('--board-annotation-file',
                      dest = 'board_annotation_file',
                      default = 'annotation2D.txt')
    parser.add_option('--label-annotation-file',
                      dest = 'label_annotation_file',
                      default = 'labels_25ms.csv')
    parser.add_option('--input-dir',
                      dest  = 'input_dir',
                      default = '/homes/abarch/hapticexp/code/expmodel/featuredata')

    (options, args) = parser.parse_args(sys.argv)
    
    # data extraction parameters  
    board = [1,1]


    # reading data from files
    board_annotation_file=os.path.join(options.input_dir, 
                                       options.board_annotation_file)
    distances= parseFile(os.path.join(options.input_dir, 
                                      options.distance_file),  
                         board_annotation_file, board, range(2,12)) 
    fingers= parseFile(os.path.join(options.input_dir, 
                                    options.fingerv_file), 
                       board_annotation_file, board, range(1,11))
    hand =  parseFile(os.path.join(options.input_dir, options.handv_file), 
                      board_annotation_file, board, range(1,4))
    labels = parseFile(os.path.join(options.input_dir, 
                                    options.label_annotation_file), 
                       board_annotation_file, board, range(2, 7))

    # put data togeather with its labels
    X,Y = composeFeatures()

    

    # data processing  for rbms
    test_size = 0.2
    
    #scaler to 1 variance
    scaler = preprocessing.StandardScaler(with_mean=False)
    X=scaler.fit_transform(X)

    # 0-1 scaling
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=test_size,
                                                        random_state=0)
    
    #models    
    rbm = BernoulliRBM(random_state=0, verbose=True)
    
    multilabel=  OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, 
                                                        random_state=0))
    


    classifier = Pipeline(steps=[('rbm', rbm), ('multilabel', multilabel)])
       
    ###############################################################################
    # Training
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # rbm components
    rbm.n_components = 20
      
    # Training Pipeline
    classifier.fit(X_train, Y_train)

    multilabel_classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, 
                                                        random_state=0))
       
    multilabel_classifier.fit(X_train,Y_train) 
    ###############################################################################
    # Evaluation
                                                 
 
    print()
    print("classification using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_train,
            classifier.predict(X_train))))
    
    print("classification on raw features:\n%s\n" % (
        metrics.classification_report(
            Y_train,
            multilabel_classifier.predict(X_train))))            
      
                                                             
    

