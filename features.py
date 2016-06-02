
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


                
