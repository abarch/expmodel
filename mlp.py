import sys
import optparse
import os
import numpy as np


#sklearn imports
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split   

#local imports 
from features import parseFile
from features import composeFeatures 



#keras imports 
from keras.models import Sequential
from keras.layers.core import Dense, Activation

        


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
    X,Y = composeFeatures(distances,fingers,hand,labels)
    print X.shape
    print Y.shape
    

    #preprocessing
    #scaler to 1 variance
    scaler = preprocessing.StandardScaler(with_mean=False)
    X=scaler.fit_transform(X)

    
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=test_size,
                                                        random_state=0)
        
    verbose = 1 
    batch_size =  32
    nb_epoch=1000
    dim_input = X.shape[1]
    dim_output = Y.shape[1]

    model = Sequential()          
    model.add(Dense(input_dim=dim_input, output_dim=20))
    model.add(Activation("sigmoid"))
    model.add(Dense(input_dim=100, output_dim=dim_output))
    model.add(Activation("softmax"))

    model.compile(loss='mse', optimizer='adadelta')
    
    model.fit(X_train, Y_train, batch_size=batch_size, show_accuracy=True,
              nb_epoch=nb_epoch) # , nb_epoch=5, batch_size=32)
    loss,accuracy  = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=verbose)
    print loss, accuracy 
    
    prediction = model.predict(X_test)
    for i in range(len(X_test))[500:520]:
        print "%s --> %s" % (str(Y_test[i]), str(prediction[i]))
        
# adadelta, rmsprop, sgd
# classification anstatt linear
# softmax 
