import math
import random

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from itertools import combinations

def perturb(REAL_DIST,USERNUM, p,q):
    tmp_perturbed_count_1 = np.copy(REAL_DIST)
    est_count = np.random.binomial(tmp_perturbed_count_1, p)
    tmp_perturbed_count_0 = USERNUM - np.copy(REAL_DIST)
    est_count += np.random.binomial(tmp_perturbed_count_0,q)
    return est_count

def perturb_returnrepresentation(domain, USERNUM, p,q,RealUserRepresentation):
    #print(RealUserRepresentation[0])
    #tmpvalues = []
    for k in range(len(RealUserRepresentation)):
        EachRepresentation = RealUserRepresentation[k]
        noiserepresentation = []
        v = np.zeros(domain, np.int32)
        for i in range(len(EachRepresentation)):
            v[EachRepresentation[i]] = 1
            for j in range(domain):
                if v[j] == 1:
                    p_sample = np.random.random_sample()
                    if p_sample <= p:
                        noiserepresentation.append(j)
                else:
                    p_sample = np.random.random_sample()
                    if p_sample > (1-q):
                        noiserepresentation.append(j)
        RealUserRepresentation[k] = noiserepresentation
        #tmpvalues.append(v)
    #print(RealUserRepresentation[0])
    est_count = np.zeros(domain,np.int32)
    for tmp in RealUserRepresentation:
        for v in tmp:
            est_count[v] += 1
    #print(len(est_count))
    #print(est_count)
    return est_count

def aggregate(REAL_DIST, USERNUM, p,q):
    est_count = np.copy(REAL_DIST)
    a = 1.0 / (p - q)
    b = q * USERNUM / (p -q)
    est_count = a * est_count - b
    return est_count

def perturb_aggregate(REAL_DIST,USERNUM, p,q):
    tmp_perturbed_count_1 = np.copy(REAL_DIST)
    est_count = np.random.binomial(tmp_perturbed_count_1, p)
    tmp_perturbed_count_0 = USERNUM - np.copy(REAL_DIST)
    est_count += np.random.binomial(tmp_perturbed_count_0,q)
    a = 1.0 / (p - q)
    b = q * USERNUM / (p - q)
    est_count = a * est_count - b
    return est_count


def deattackfalse(tmpdataset,p,q,NUM, domain,Targets,RealUserNUM):

    #SAMPLENUM = int(len(tmpdataset)/ 5)   #sample  1/5 data

    #dataset =  random.sample(tmpdataset, SAMPLENUM)

    dataset = tmpdataset


    FID = set()

    #OBJ = Targets
    #LEN = len(OBJ)

    OBJ =  range(domain)
    MAXLEN = 10
    LEN = len(OBJ)
    if LEN > MAXLEN:
        LEN = MAXLEN

    for i in range(1,LEN+1):
        print('i:', i)
        for tup in combinations(OBJ, i):
            count = 0
            indexarray = []
            for j in range(len(dataset)):
                data = dataset[j]
                flag = True
                for item in tup:
                    if item not in data:
                        flag = False
                        break
                if flag:
                    count = count + 1
                    indexarray.append(j)
            threshold = len(dataset) * p * q ** (i - 1) - 10 * math.sqrt(
                len(dataset) * p * q ** (i - 1) * (1 - p * q ** (i - 1)))
            if count > threshold:
                for id in indexarray:
                    FID.add(id)

    #print(FID)
    print('FID Size:', len(FID))
    count = 0
    for item in FID:
        if item < RealUserNUM:
            count += 1
    print('False Positive:', count)
    # FID = []

    DIST = np.zeros(domain, dtype=np.int32)
    for i in range(len(dataset)):
        data = dataset[i]
        if i not in FID:
            for item in data:
                DIST[item] += 1
    ratio = len(tmpdataset) / len(dataset)
    DIST = DIST * ratio
    print('DIST[0:20]:', DIST[0:20])

    NewUserNUM = NUM - len(FID)

    return NewUserNUM, DIST


def deattack22222(tmpdataset,p,q,NUM, domain,Targets,RealUserNUM):

    SAMPLERATE = 1
    dataset = []
    for i in range(len(tmpdataset)):
        if i % SAMPLERATE == 0:
            dataset.append(tmpdataset[i])
    print('sampled dataset size:', len(dataset))



    FID = set()
    start = 1
    maxlen = 11
    tmparr = list(range(domain))
    #arr = tmparr

    print('Targets:',Targets)
    arr = Targets.copy()
    #tmplen = 2*len(arr)
    tmplen = 2 * len(arr) -5
    random.shuffle(tmparr)
    print(tmparr)
    for item in tmparr:
        if item not in arr:
            arr.append(item)
            if len(arr) == tmplen:
                break
    print('arr:',arr)


    samplecombine = []
    samplecombine.append(Targets)


    start = 1
    maxlen = 11

    for i in range(start,maxlen):
         print('$$$$$$$,',i)
         #if i > 5: break
         for tup in combinations(arr, i):
        # for tup in samplecombine:
             count = 0
             indexarray = []
             for j in range(len(dataset)):
                 data = dataset[j]
                 flag = True
                 for item in tup:
                     if item not in data:
                         flag = False
                         break
                 if flag:
                     count = count + 1
                     indexarray.append(j)
             threshold = len(dataset) * p * q**(i-1) - 10 * math.sqrt(len(dataset)*p * q**(i-1)*(1-p*q**(i-1)))
             if count > threshold:
                 for id in indexarray:
                     FID.add(id)
                 #print(FID)
    print(FID)
    print('FID Size:', len(FID))
    count = 0
    for item in FID:
        if item < RealUserNUM:
            count += 1
    print('False Positive:', count)
    #FID = []

    DIST = np.zeros(domain, dtype=np.int32)
    for i in range(len(dataset)):
        data = dataset[i]
        if i not in FID:
            for item in data:
                DIST[item] += 1
    ratio = len(tmpdataset)/len(dataset)
    DIST = DIST * ratio
    print('DIST[0:20]:',DIST[0:20])

    NewUserNUM = NUM - len(FID)

    return NewUserNUM,DIST

def deattack(tmpdataset,p,q,NUM, domain,Targets,RealUserNUM):

    SAMPLERATE = 1
    dataset = []
    for i in range(len(tmpdataset)):
        if i % SAMPLERATE == 0:
            dataset.append(tmpdataset[i])
    print('sampled dataset size:', len(dataset))



    FID = set()
    start = 1
    maxlen = 11
    tmparr = list(range(domain))
    #arr = tmparr

    print('Targets:',Targets)
    arr = Targets.copy()
    #tmplen = 2*len(arr)
    tmplen = 2 * len(arr) -5
    random.shuffle(tmparr)
    print(tmparr)
    for item in tmparr:
        if item not in arr:
            arr.append(item)
            if len(arr) == tmplen:
                break
    print('arr:',arr)
    start = 10
    maxlen = 11

    samplecombine = []
    samplecombine.append(Targets)

    for i in range(start,maxlen):
         print('$$$$$$$,',i)
         #if i > 5: break
         #for tup in combinations(arr, i):
         for tup in samplecombine:
             count = 0
             indexarray = []
             for j in range(len(dataset)):
                 data = dataset[j]
                 flag = True
                 for item in tup:
                     if item not in data:
                         flag = False
                         break
                 if flag:
                     count = count + 1
                     indexarray.append(j)
             threshold = len(dataset) * p * q**(i-1) - 10 * math.sqrt(len(dataset)*p * q**(i-1)*(1-p*q**(i-1)))
             if count > threshold:
                 for id in indexarray:
                     FID.add(id)
                 #print(FID)
   # print(FID)
    print('FID Size:', len(FID))
    count = 0
    for item in FID:
        if item < RealUserNUM:
            count += 1
    print('False Positive:', count)
    #FID = []

    DIST = np.zeros(domain, dtype=np.int32)
    for i in range(len(dataset)):
        data = dataset[i]
        if i not in FID:
            for item in data:
                DIST[item] += 1
    ratio = len(tmpdataset)/len(dataset)
    DIST = DIST * ratio
    print('DIST[0:20]:',DIST[0:20])

    NewUserNUM = NUM - len(FID)

    return NewUserNUM,DIST
    # FID = set()
    # maxlen = 11
    # arr = range(domain)
    # print('arr:',arr)
    # for i in range(1,maxlen):
    #     print('$$$$$$$,',i)
    #     if i > 5: break
    #     for tup in combinations(arr, i):
    #         count = 0
    #         indexarray = []
    #         for j in range(len(dataset)):
    #             data = dataset[j]
    #             flag = True
    #             for item in tup:
    #                 if item not in data:
    #                     flag = False
    #                     break
    #             if flag:
    #                 count = count + 1
    #                 indexarray.append(j)
    #         threshold = len(dataset) * p * q**(i-1) - 10 * math.sqrt(len(dataset)*p * q**(i-1)*(1-p*q**(i-1)))
    #         if count > threshold:
    #             for id in indexarray:
    #                 FID.add(id)
    #     print(FID)
    # print(FID)
    # print('FID Size:', len(FID))

    # te = TransactionEncoder()
    # te_ary = te.fit(dataset).transform(dataset)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # minisupport = p*(q**4)
    # frequentpattern = fpgrowth(df, min_support=minisupport, use_colnames=True)
    # print(frequentpattern)



# def perturb(X, n, p, q):
#     Y = np.zeros(n, dtype=np.int32)
#     for i in range(n):
#         v = X[i]
#         y = v
#         p_sample = np.random.random_sample()
#         if p_sample > p - q:
#             # perturb
#             y = np.random.randint(0, g)
#         Y[i] = y
#     return Y
#
# def aggregate(Y,domain,n,p,q):
#     ESTIMATE_DIST = np.zeros(domain,dtype=np.int32)
#     for i in range(n):
#         ESTIMATE_DIST[Y[i]] += 1
#     #a = 1.0 / n
#     #b = q
#     a = 1.0
#     b = q * n
#     print('b:',b)
#     ESTIMATE_DIST = (a * ESTIMATE_DIST - b) / (p-q)
#     return ESTIMATE_DIST


def error_metric(REAL_DIST,ESTIMATE_DIST,domain):
    abs_error = 0.0
    for x in range(domain):
        # print REAL_DIST[x], ESTIMATE_DIST[x]
        abs_error += np.abs(REAL_DIST[x] - ESTIMATE_DIST[x]) ** 2
    return abs_error / domain