from compiler.ast import flatten
import numpy as np

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

def composeFeatures (distances, fingers,hand,labels):
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
        

    def selectFinger(finger_nb):
        zeros= [0]*16
        zeros[int(labeli[finger_nb])]=1
        return zeros
    
        
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
                labeli = selectFinger(1) 
                
                labeling += [labeli]
            
                alldata +=[map(float, features)]
                
    X = np.array(alldata)
    Y = np.array(labeling) 
    
    return X,Y 
                

def evaluateWith(X_test,Y_test, model):
    def stats(prediction):

        def classequals(label, pred):
            label = list(label)
            pred = list(pred)
            
            first_max = (pred.index(max(pred))== label.index(max(label)))
            pred[pred.index(max(pred))]= min(pred)
            second_max = (pred.index(max(pred))== label.index(max(label)))
                 
            return (first_max or second_max) 
        
        correct=0
        overall=len(X_test)
            
        for i in range(overall):
        #print "%s --> %s" % (str(Y_test[i]), str(prediction[i]))
            if classequals(Y_test[i], prediction[i]): 
                correct+=1
        return correct,overall

    
    loss,accuracy  = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=verbose)
    print "loss=", loss, " accuracy=",  accuracy 
    
    prediction = model.predict(X_test)
    correct,overall= stats(prediction)
    print "correct=", correct, " overall=", overall
