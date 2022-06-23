import math
from itertools import combinations

import numpy as np
import xxhash

import scipy.special as sc

def perturb(X, n, g, p, q):
    Y = np.zeros(n,dtype=np.int32)
    for i in range(n):
        v = X[i]
        x = (xxhash.xxh32(str(v), seed=i).intdigest() % g)
        y = x
        p_sample = np.random.random_sample()
        # the following two are equivalent
        # if p_sample > p:
        #     while not y == x:
        #         y = np.random.randint(0, g)
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        Y[i] = y
    return Y

def perturbWithSeeds(X, n, g, p, q, seeds):
    Y = np.zeros(n,dtype=np.int32)
    for i in range(n):
        v = X[i]
        x = (xxhash.xxh32(str(v), seed=seeds[i]).intdigest() % g)
        y = x
        p_sample = np.random.random_sample()
        # the following two are equivalent
        # if p_sample > p:
        #     while not y == x:
        #         y = np.random.randint(0, g)
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        Y[i] = y
    return Y

def EncodewithSelectedSeeds(X, n, g, p, q, seeds):
    Y = np.zeros(n,dtype=np.int32)
    for i in range(n):
        v = X[i]
        x = (xxhash.xxh32(str(v), seed=seeds[i]).intdigest() % g)
        y = x
        Y[i] = y
    return Y

def aggregate(Y,domain,n,g,p,q):
    ESTIMATE_DIST = np.zeros(domain,dtype=np.int32)
    for i in range(n):
        for v in range(domain):
            if Y[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                ESTIMATE_DIST[v] += 1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    ESTIMATE_DIST = a * ESTIMATE_DIST - b
    return ESTIMATE_DIST

def aggregateWithSeeds(Y,domain,n,g,p,q,seeds):
    ESTIMATE_DIST = np.zeros(domain,dtype=np.int32)
    for i in range(n):
        for v in range(domain):
            if Y[i] == (xxhash.xxh32(str(v), seed=seeds[i]).intdigest() % g):
                ESTIMATE_DIST[v] += 1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    ESTIMATE_DIST = a * ESTIMATE_DIST - b
    return ESTIMATE_DIST

def error_metric(REAL_DIST,ESTIMATE_DIST,domain):
    abs_error = 0.0
    for x in range(domain):
        # print REAL_DIST[x], ESTIMATE_DIST[x]
        abs_error += np.abs(REAL_DIST[x] - ESTIMATE_DIST[x]) ** 2
    return abs_error / domain

def deattack(tempdataset,seeds, p,q,NUM, domain,Targets,RealUserNUM,g):
    print("*********deattack********")
    dataset = []
    for i in range(NUM):
        EachRecord = []
        for v in range(domain):
            if tempdataset[i] == (xxhash.xxh32(str(v), seed=seeds[i]).intdigest() % g):
                EachRecord.append(v)
        dataset.append(EachRecord)

    FID = set()
    start = 1
    maxlen = 9

    samplecombine = Targets

    for i in range(start, maxlen):
        print('$$$$$$$,', i)
        # if i > 5: break
        #calculate tau_z
        tau_z = -1
        for tmptau in range(1, NUM, 5):
            if sc.betainc(tmptau, NUM - tmptau + 1,q ** (i - 1))  / sc.betainc(tmptau, NUM - tmptau + 1,1)  < 0.01:
                tau_z = tmptau
                break
        print("tau_z is: ", tau_z)

        if tau_z == -1:
            continue

        for tup in combinations(samplecombine, i):
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

            threshold = tau_z

            if count > threshold:
                for id in indexarray:
                    FID.add(id)

   # print(FID)
    print('FID Size:', len(FID))
    count = 0
    for item in FID:
        if item < RealUserNUM:
            count += 1
    print('False Positive:', count)
    # FID = []



    DeAttackNewUserNUM = NUM - len(FID)
    DeAttack_UserRepresentation = []
    DeAttack_SEEDS = []
    for i in range(len(tempdataset)):
        data = tempdataset[i]
        seed = seeds[i]
        if i not in FID:
            DeAttack_UserRepresentation.append(data)
            DeAttack_SEEDS.append(seed)
    return DeAttackNewUserNUM, DeAttack_UserRepresentation, DeAttack_SEEDS