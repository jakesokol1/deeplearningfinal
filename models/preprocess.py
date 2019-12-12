import numpy as np
from functools import reduce
import itertools
from .subject_graph import SubjectGraph


#tuple_line: takes a line and casts it as a list, as well as
def tuple_line(line):
    splitter = [x.strip() for x in line.split(',')]
    return splitter
    #splitter = ['sub','emotion','x1','y1','rt','study']

def listify(filename): #reads .csv file to create list of data
    with open(filename) as f:
        lines = f.read().splitlines()
    lines = lines[1:]#ignores labels of first line
    lines = [tuple_line(x) for x in lines]
    return lines
#line = 'sub,emotion,X1,Y1,RT,study' each sub has 20 lines to themselves
#incorporate: appends two items to each line, first item is the data on subject Alexithymia
#second item is subject's Clinical Depression
def incorporate(bfile, st1, st3):
    lines = listify(bfile)
    s1 = listify(st1)
    s3 = listify(st3)
    for _, y in enumerate(lines):
        if y[5]=='study1':
            for _, j in enumerate(s1):
                if y[0] == j[0] and len(y) < 8:
                    y.append(j[14])
                    y.append(None)
                elif len(y) >= 8:
                    continue
        elif y[5]=='study3':
            for _,j in enumerate(s3):
                if y[0]==j[0] and len(y) < 8:
                    y.append(j[25])
                    y.append(j[7])
                elif len(y) >= 8:
                    continue
        else:
            y.append(None)
            y.append(None)
    return lines
#sample line from incorporate: ['95', 'calm', '0', '0', '14.823', 'study4', None, None]
#TODO: make a dict for each subject, one to give x1,y1 given a word, another for alexithymia, third for depression

def xy_dictify(subjects):
    xybig_dict = []
    for _, sub in enumerate(subjects):
        diction = dict()
        for i in range(len(sub)):
            diction[int(sub[i][1])] = [int(sub[i][2]), int(sub[i][3])]
        xybig_dict.append(diction)
    return xybig_dict
#distbig_dict is a dictionary list with two-word keys, with keys (w1[x1]-w2[x1],w1[y1]-w2[y1])
def dist_dictify(xydict):
    dist_dict = dict()
    word_list = list(xydict.keys())
    xy_list = list(xydict.values())
    wordpairlist = list(itertools.permutations(word_list, 2))
    tuplepairlist = list(itertools.permutations(xy_list, 2))
    coord_distlist = []
    for j, tpair in enumerate(tuplepairlist):
        dx = float(tpair[0][0]) - float(tpair[1][0])
        dy = float(tpair[0][1]) - float(tpair[1][1])
        d = [dx, dy]
        coord_distlist.append(d)
    for i, pair in enumerate(wordpairlist):
        dist_dict[pair] = coord_distlist[i]
    return dist_dict

def alex_dictify(subjects):
    alexbig_dict=[]
    for _, sub in enumerate(subjects):
        diction = dict()
        for i in range(len(sub)):
            val = sub[i][6]
            if val == '"Alexithymia"':
                diction[int(sub[i][1])] = 1
            elif val == '"No_Alexithymia"':
                diction[int(sub[i][1])] = 0
            else:
                assert(val == '"Possible_Alexithymia"' or val == 'NA' or val is None)
                diction[int(sub[i][1])] = None
        alexbig_dict.append(diction)
    return alexbig_dict

def depression_dictify(subjects):
    depbig_dict=[]
    for _, sub in enumerate(subjects):
        diction = dict()

        for i in range(len(sub)):
            val = sub[i][7]
            if val == '"Clinically_Depressed"':
                diction[int(sub[i][1])] = 1
            elif val == '"Healthy_Controls"':
                diction[int(sub[i][1])] = 0
            else:
                assert(val is None)
                diction[int(sub[i][1])] = None
        depbig_dict.append(diction)
    return depbig_dict

def numberify(data):#relabels emotion words with integers 0-19 specifically from incorporate/listify output
    e = ['annoyed', 'relaxed', 'enthusiastic', 'calm', 'disappointed', 'aroused', 'neutral', 'sluggish', 'peppy', 'quiet', 'still', 'surprised', 'sleepy', 'nervous', 'afraid', 'satisfied', 'disgusted', 'angry', 'happy', 'sad']
    for _, d in enumerate(data):
        for j, k in enumerate(e):
            if d[1] == k:
                d[1] = j
            else:
                continue
    return data

def convert_to_depression_graph(subject_dicts, dist_dicts, depression_dicts):
    """
    Converts a subject dict object into a manageable SubjectGraph instance for
    dgl graph creation.

    :parameter subject_dict: dictionary representing subjects answers
    """

    subject_graphs = []

    for i in range(len(subject_dicts)):

        subject_dict = subject_dicts[i]
        dist_dict = dist_dicts[i]
        depression_dict = depression_dicts[i]
        label = None

        if depression_dict[0] is None:
            continue
        else:
            label = depression_dict[0]

        nodes = []
        for i in range(20):
            nodes.append(subject_dict[i])

        edges = []
        for i in range(20):
            for j in range(20):
                if i != j:
                    edges.append(((i, j), dist_dict[(i, j)]))
        subject_graphs.append(SubjectGraph(nodes=nodes, edges=edges, label=label))

    return subject_graphs


def convert_to_alex_graph(subject_dicts, dist_dicts, alex_dicts):
    """
    Converts a subject dict object into a manageable SubjectGraph instance for
    dgl graph creation.

    :parameter subject_dict: dictionary representing subjects answers
    """

    subject_graphs = []

    for i in range(len(subject_dicts)):

        subject_dict = subject_dicts[i]
        dist_dict = dist_dicts[i]
        alex_dict = alex_dicts[i]
        label = None

        if alex_dict[0] is None:
            continue
        else:
            label = alex_dict[0]

        nodes = []
        for i in range(20):
            nodes.append(subject_dict[i])

        edges = []
        for i in range(20):
            for j in range(20):
                if i != j:
                    edges.append(((i, j), dist_dict[(i, j)]))
        subject_graphs.append(SubjectGraph(nodes=nodes, edges=edges, label=label))

    return subject_graphs
#big duct tape energy
def get_data(condition):#f1-emotionalclassificationdata.csv f2-study1 f3-study3

    f1 = 'data/EmotionClassificationData.csv'
    f2 = 'data/Study1_Emotion_Data.csv'
    f3 = 'data/Study3_Emotion_Data.csv'

    bigdata = incorporate(f1,f2,f3)
    bigdata = numberify(bigdata)
    subject_bd = []
    for i in range(int(len(bigdata)/20)):
        dingus = np.array(bigdata[i*20:i*20 + 20])
        dingus.flatten
        dingus = tuple([tuple(row) for row in dingus])
        subject_bd.append(dingus) #subject_bd has an element which corresponds to a single subjects data
    xybig_dict = xy_dictify(subject_bd)
    alexbig_dict = alex_dictify(subject_bd)
    depbig_dict = depression_dictify(subject_bd)
    distbig_dict = []
    for _, xy in enumerate(xybig_dict):
        distbig_dict.append(dist_dictify(xy))

    if condition == 'depression':
        subject_graphs = convert_to_depression_graph(xybig_dict, distbig_dict, depbig_dict)
    elif condition == 'alex':
        subject_graphs = convert_to_alex_graph(xybig_dict, distbig_dict, depbig_dict)
    else:
        raise ValueError('Invalid condition')

    partition = .1 * len(subject_graphs)
    subject_graphs_train = subject_graphs[partition:]
    subject_graphs_test = subject_graphs[:partition]

    return (subject_graphs_train, subject_graphs_test)


#CRASH COURSE ON WHAT IS GOING ON
#each dictionary list has length = num of subjects
#each dictionary list is ordered the same way, meaning the nth dictionary in any list
#corresponds to information about the same subject
#keys for each dictionary (in no particular order):
# annoyed, relaxed, enthusiastic, calm, disappointed
# aroused, neutral, sluggish, peppy, quiet
# still, surprised, sleepy, nervous, afraid
# satisfied, disgusted, angry, happy, sad

#xybig_dict
# given a key, the value is : (x, y)
#x,y are both in a range [-250, 250]

#distbig_dict
# keys are of type (word1, word2)
#value gives (word1(x)-word2(x), word1(y)-word2(y))

#depbig_dict
# given a key, the value is either 1, 0, or None
#1 = Clinically Depressed, 0 = Healthy Control, None = info not available

#alexbig_dict
# given a key, the value is either "Alexithymia", "Possible_Alexithymia", "No_Al
