import random
from itertools import combinations
import numpy as np
import OLH as olh
import OUE as oue
import KRR as krr
import math
import xxhash
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

ZIPF = 'zipf'
UNIFORM = 'uniform'
FIRE = 'Fire'
IPUMS = 'IPUMS'

#USERNUM = 1000000
#domain = 1024

#USERNUM = 62500 * 2
#domain = 64 * 2

#USERNUM = 100000
#domain = 100

#USERNUM = 125000
#domain = 128

USERNUM = 500000
domain = 128


KRR = 'KRR'
OUE = 'OUE'
OLH = 'OLH'

RPA = 'RPA'
RIA = 'RIA'
MGA = 'MGA'

NORM = 'NORM'
DETECT = 'DETECT'
BOTH  = 'BOTH'
ALL  = 'ALL'

Beta = 0.05
#Beta = 0.05
#Beta = 0.1
#NumofTarget = 1
NumofTarget = 5
F_t = 0.01
epsilon = 1.0
NumofAttacks = 0


def EnumTargetsCloseToFt(arr, r, target):
    arr2 = arr.copy()
    indexofarr = sorted(range(len(arr2)), key=lambda k: arr2[k])

    TmpTargets = []
    flag = True
    i = 0
    j = i+1
    while flag:
        arr_i = []
        Targets_i = []
        for k1 in range(i,i+r):
            arr_i.append(arr[indexofarr[k1]])
            Targets_i.append(indexofarr[k1])

        arr_j = []
        Targets_j = []
        for k2 in range(j, j + r):
            arr_j.append(arr[indexofarr[k2]])
            Targets_j.append(indexofarr[k2])

        if abs(sum(arr_i)/USERNUM - F_t) < abs(sum(arr_j)/USERNUM - F_t):
            TmpTargets = Targets_i
            print('IDS: ',arr_i )
            print('sum(IDS): ', sum(arr_i)/USERNUM )
            return TmpTargets

        i = i + 1
        j = i + 1

    return TmpTargets

'''
def EnumTargetsCloseToFt(arr, r, target):
    if r == 10:
        return EnumTargetsCloseToFt2(arr,r)
    dic = {}
    for tup in combinations(arr, r):
        try:
            dic[np.absolute(np.sum(tup) - target)].append(str(tup))
        except KeyError:
            dic[np.absolute(np.sum(tup) - target)] = [tup]
    print(dic[min(dic.keys())])
    return dic[min(dic.keys())]
def EnumTargetsCloseToFt2(arr2,k):
    arr = arr2.copy()
    arr.sort()
    print('sum of arr[:k]:',sum(arr[:k])/USERNUM)
    print(arr[:k])
    return [arr[:k]]
'''

def originaldata_generation(dis):
    # Use a breakpoint in the code line below to debug your script.
    if dis == ZIPF:
        tmpsum = 0
        for i in range(1, domain + 1):
            tmpsum += 1.0 / i
        tmpcount = np.zeros(domain + 1, dtype=np.int32)
        for i in range(1, domain + 1):
            tmpcount[i] = int(round(USERNUM * 1.0 / tmpsum * (1.0 / i)))
        print(tmpcount[12], tmpcount[13], tmpcount[14])
        dataset = np.zeros(USERNUM, dtype=np.int32)
        tmpid = 1
        for i in range(USERNUM):
            if tmpcount[tmpid] > 0:
                dataset[i] = tmpid - 1
                tmpcount[tmpid] -= 1
                if tmpcount[tmpid] == 0 and tmpid < domain:
                    tmpid += 1
            elif tmpid == domain and tmpcount[tmpid] == 0:
                # dataset[i] = np.random.randint(domain)
                dataset[i] = domain - 1
        # np.savetxt("zipfdata.txt", dataset, fmt="%d", delimiter=",")
    elif dis == UNIFORM:
        dataset = np.zeros(USERNUM, dtype=np.int32)
        for i in range(USERNUM):
            dataset[i] = np.random.randint(domain)
        # np.savetxt("uniformdata.txt", dataset, fmt="%d", delimiter=",")
    elif dis == FIRE:
        f = open("Fire.csv")
        line = f.readline()  # skip the header
        line = f.readline()
        unitIDList = []
        while line:
            tmpstr_array = line.split()
            str_array = tmpstr_array[0].split(',')
            if str_array[3] == 'Alarms':
                unitIDList.append(str_array[1])
            line = f.readline()
        f.close()
        # print(len(unitIDList))
        # print(unitIDList[1:20])
        unitIDSetList = list(set(unitIDList))
        for i in range(len(unitIDList)):
            id = unitIDList[i]
            unitIDList[i] = unitIDSetList.index(id)
        # print(unitIDSetList)
        # print(unitIDList[1:20])
        dataset = unitIDList
    elif dis == IPUMS:
        f = open("IPUMS.dat")
        line = f.readline()
        unitIDList = []
        while line:
            str = line.strip()
            if str != '0000':
                unitIDList.append(str)
            line = f.readline()
        f.close()

        print(set(unitIDList))
        unitIDSetList = list(set(unitIDList))

        for i in range(len(unitIDList)):
            id = unitIDList[i]
            unitIDList[i] = unitIDSetList.index(id)
        # print(unitIDSetList)
        # print(unitIDList[1:20])
        dataset = unitIDList
    return dataset



def run(perturbway, attackway, original_dataset, epsilon,deattackway):
    global NumofAttacks
    ESTIMATE_DIST_Attacked = []
    ESTIMATE_DIST = []
    Targets = []
    UserRepresentation = []
    # default
    NewUserNUM = USERNUM + NumofAttacks


    if perturbway == OLH:
        g = int(round(math.exp(float(epsilon)))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / (math.exp(epsilon) + g - 1)
        print('g,p,q:', g,p,q)
        print('perturbway:', perturbway)
        seeds = np.random.randint(0, len(original_dataset), len(original_dataset))
        perturbed_original_dataset = olh.perturbWithSeeds(original_dataset, len(original_dataset), g, p, q,seeds)
        print('perturbed_original_dataset[0:20]: ', perturbed_original_dataset[0:20])
        print('UserNum:', USERNUM)
        REAL_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(USERNUM):
            REAL_DIST[original_dataset[i]] += 1
        ESTIMATE_DIST = olh.aggregateWithSeeds(perturbed_original_dataset, domain, USERNUM, g, p, q,seeds)
        ERROR = olh.error_metric(REAL_DIST, ESTIMATE_DIST, domain)
        print('REAL_DIST[0:20]:', REAL_DIST[0:20])
        print('ESTIMATE_DIST[0:20]: ', ESTIMATE_DIST[0:20])
        print('ERROR:', ERROR)

        Copy_original_dataset = perturbed_original_dataset.copy()
        '''
                Targets = []
                TargetValues = EnumTargetsCloseToFt(REAL_DIST, NumofTarget, F_t * USERNUM)[0]
                print('TargetValues:', TargetValues)
                for v in TargetValues:
                    tmp = 0
                    for f in REAL_DIST:
                        flag = False
                        if f == v:
                            Targets.append(tmp)
                            flag = True
                        tmp += 1
                        if flag:
                            break
                print('Targets:', Targets)
                '''

        Targets = EnumTargetsCloseToFt(REAL_DIST, NumofTarget, F_t * USERNUM)

        print('Targets:', Targets)

        if len(Targets) != NumofTarget:
            print('Error!!! len(Targets) != NumofTarget')

        print('REAL_DIST[Targets[0]]:', REAL_DIST[Targets[0]])

        UserRepresentation = []
        UserSEEDS = []


        # beta = m / (n+m)
        NumofAttacks = int(Beta * USERNUM / (1 - Beta))
        print('NumofAttacks:', NumofAttacks)

        if attackway == RPA:
            # generate poisoning data
            poisoning_data  = np.random.randint(0, domain, NumofAttacks)
            poisoning_data_seeds = np.random.randint(0, domain, NumofAttacks)
            encoded_poisoning_data = olh.EncodewithSelectedSeeds(poisoning_data, len(poisoning_data), g, p, q,
                                                                   poisoning_data_seeds)
            for i in range(NumofAttacks):
                Copy_original_dataset = np.append(Copy_original_dataset, encoded_poisoning_data[i])
            newseeds = []
            for i in range(len(Copy_original_dataset)):
                if i < len(original_dataset):
                    newseeds.append(seeds[i])
                else:
                    newseeds.append(poisoning_data_seeds[i - len(original_dataset)])

            UserRepresentation = Copy_original_dataset.copy()
            UserSEEDS = newseeds.copy()

            ESTIMATE_DIST_Attacked = olh.aggregateWithSeeds(Copy_original_dataset, domain, len(Copy_original_dataset), g, p, q,newseeds)
            ERROR_Attacked = olh.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        elif attackway == RIA:
            # generate poisoning data
            poisoning_data = []
            for i in range(NumofAttacks):
                poisoning_data.append(Targets[np.random.randint(len(Targets))])
            poisoning_data_seeds = np.random.randint(0, domain, NumofAttacks)
            perturbed_poisoning_data = olh.perturbWithSeeds(poisoning_data, len(poisoning_data), g, p, q,poisoning_data_seeds)
            for i in range(NumofAttacks):
                Copy_original_dataset = np.append(Copy_original_dataset, perturbed_poisoning_data[i])
            newseeds = []
            for i in range(len(Copy_original_dataset)):
                if i < len(original_dataset):
                    newseeds.append(seeds[i])
                else:
                    newseeds.append(poisoning_data_seeds[i - len(original_dataset)])

            UserRepresentation = Copy_original_dataset.copy()
            UserSEEDS = newseeds.copy()

            ESTIMATE_DIST_Attacked = olh.aggregateWithSeeds(Copy_original_dataset, domain, len(Copy_original_dataset),
                                                            g, p, q, newseeds)
            ERROR_Attacked = olh.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)

            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        elif attackway == MGA:
            # select a hash function such that 'for each fake user, we randomly sample 1,000 hash functions and
            # use the one that hashes the most target items to the same value.'
            samplesize = 1000
            count = np.zeros(samplesize, dtype=np.int32)
            maxseed = 0
            maxcount = 0
            for i in range(samplesize):
                tmpcount = np.zeros(g, dtype=np.int32)
                for j in Targets:
                    tmpcount[xxhash.xxh32(str(j), seed=i).intdigest() % g] += 1
                count[i] = max(tmpcount)
                if max(tmpcount) > maxcount:
                    maxseed = i
                    maxcount = max(tmpcount)
            print('maxseed:', maxseed)
            print('maxcount:', maxcount)
            print('count:', count)

            # generate poisoning data
            poisoning_data = []
            poisoning_data_seeds = []
            for i in range(NumofAttacks):
                poisoning_data.append(Targets[np.random.randint(len(Targets))])
                poisoning_data_seeds.append(maxseed)
            perturbed_poisoning_data = olh.EncodewithSelectedSeeds(poisoning_data, len(poisoning_data), g, p, q,poisoning_data_seeds)
            for i in range(NumofAttacks):
                Copy_original_dataset = np.append(Copy_original_dataset, perturbed_poisoning_data[i])
            newseeds= []
            for i in range(len(Copy_original_dataset)):
                if i < len(original_dataset):
                    newseeds.append(seeds[i])
                else:
                    newseeds.append(poisoning_data_seeds[i-len(original_dataset)])

            UserRepresentation = Copy_original_dataset.copy()
            UserSEEDS = newseeds.copy()

            ESTIMATE_DIST_Attacked = olh.aggregateWithSeeds(Copy_original_dataset, domain, len(Copy_original_dataset), g, p, q, newseeds)
            ERROR_Attacked = olh.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)

            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(),Targets,USERNUM)

        if attackway == RPA or  attackway == MGA:
            DeAttackNewUserNUM, DeAttack_UserRepresentation, DeAttack_SEEDS = olh.deattack(UserRepresentation, UserSEEDS, p, q, USERNUM+NumofAttacks, domain, Targets, USERNUM,g)
            #DeAttackNewUserNUM, DeAttack_UserRepresentation, DeAttack_SEEDS = olh.deattack(UserRepresentation,UserSEEDS, p, 1.0/g,USERNUM + NumofAttacks, domain, Targets, USERNUM, g)

            ESTIMATE_DIST_Attacked = olh.aggregateWithSeeds(DeAttack_UserRepresentation, domain, len(DeAttack_UserRepresentation),  g, p, q, DeAttack_SEEDS)
            ERROR_Attacked = olh.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)

            print('DetectedAbnormalUsers:', (USERNUM+NumofAttacks-DeAttackNewUserNUM))
            print('Estimated Beta:', (USERNUM+NumofAttacks-DeAttackNewUserNUM)/(USERNUM+NumofAttacks))

            print(Targets)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            print('NumofAttacks:',NumofAttacks)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / DeAttackNewUserNUM - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('After De-Attack(Detection) Gain is:', Gain1)

    elif perturbway == OUE:
        g = 2 ** domain
        p = 1.0 / 2.0
        q = 1.0 / (math.exp(epsilon) + 1)
        print('g,p,q:', g,p,q)
        print('perturbway:', perturbway)
        REAL_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(USERNUM):
            REAL_DIST[original_dataset[i]] += 1
            uservalue = []
            uservalue.append(original_dataset[i])
            UserRepresentation.append(uservalue)
        #POISION_DIST = oue.perturb(REAL_DIST, USERNUM, p, q)
        POISION_DIST = oue.perturb_returnrepresentation(domain,USERNUM, p,q,UserRepresentation)
        ESTIMATE_DIST = oue.aggregate(POISION_DIST,USERNUM, p,q)
        ERROR = oue.error_metric(REAL_DIST, ESTIMATE_DIST, domain)
        print('REAL_DIST[0:20]:', REAL_DIST[0:20])
        print('ESTIMATE_DIST[0:20]: ', ESTIMATE_DIST[0:20])
        print('ERROR:', ERROR)
        '''
                Targets = []
                TargetValues = EnumTargetsCloseToFt(REAL_DIST, NumofTarget, F_t * USERNUM)[0]
                print('TargetValues:', TargetValues)
                for v in TargetValues:
                    tmp = 0
                    for f in REAL_DIST:
                        flag = False
                        if f == v:
                            Targets.append(tmp)
                            flag = True
                        tmp += 1
                        if flag:
                            break
                print('Targets:', Targets)
                '''

        Targets = EnumTargetsCloseToFt(REAL_DIST, NumofTarget, F_t * USERNUM)
        print('Targets:', Targets)
        if len(Targets) != NumofTarget:
            print('Error!!! len(Targets) != NumofTarget')
        print('REAL_DIST[Targets[0]]:', REAL_DIST[Targets[0]])
        NumofAttacks = int(Beta * USERNUM / (1 - Beta))
        print('NumofAttacks:', NumofAttacks)
        if attackway == RPA:
            POISION_DIST_TMP = POISION_DIST.copy()
            for i in range(NumofAttacks):
                #for item in Targets:
                noiserepresentation = []
                for item in range(domain):
                    p_sample = np.random.random_sample()
                    if p_sample > 0.5:
                        POISION_DIST_TMP[item] += 1
                        noiserepresentation.append(item)
                UserRepresentation.append(noiserepresentation)
            print('POISION_DIST_TMP:')
            print(POISION_DIST_TMP)
            ESTIMATE_DIST_Attacked = oue.aggregate(POISION_DIST_TMP, USERNUM+NumofAttacks, p,q)
            ERROR_Attacked = oue.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        elif attackway == RIA:
            ESTIMATE_DIST_TMP = REAL_DIST.copy()
            for i in range(NumofAttacks):
                item = Targets[np.random.randint(len(Targets))]
                ESTIMATE_DIST_TMP[item] += 1
            ESTIMATE_DIST_Attacked = oue.perturb_aggregate(ESTIMATE_DIST_TMP, USERNUM + NumofAttacks, p, q)
            ERROR_Attacked = oue.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        elif attackway == MGA:
            POISION_DIST_TMP = POISION_DIST.copy()
            for i in range(NumofAttacks):
                noiserepresentation = []
                for item in Targets:
                    POISION_DIST_TMP[item] += 1
                    noiserepresentation.append(item)
                NumOfAdditionalOne = int(p+(domain-1)*q-NumofTarget)
                TMPIDList = []
                for j in range(domain):
                    if j not in Targets:
                        TMPIDList.append(j)
                SelectedUntargetID = np.random.choice(len(TMPIDList),NumOfAdditionalOne,replace=False)
                for j in range(NumOfAdditionalOne):
                    item = TMPIDList[SelectedUntargetID[j]]
                    POISION_DIST_TMP[item] += 1
                    noiserepresentation.append(item)
                UserRepresentation.append(noiserepresentation)
            print('POISION_DIST_TMP[0:20]:')
            print(POISION_DIST_TMP[0:20])
            ESTIMATE_DIST_Attacked = oue.aggregate(POISION_DIST_TMP, USERNUM + NumofAttacks, p, q)
            ERROR_Attacked = oue.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        '''
        # ESTIMATE_DIST_Attacked = ESTIMATE_DIST_Attacked * 1.0 / (USERNUM + NumofAttacks)
        ESTIMATE_DIST_Attacked = ESTIMATE_DIST_Attacked * 1.0 / NewUserNUM
        MinV = min(ESTIMATE_DIST_Attacked)
        SumV = 0
        for i in ESTIMATE_DIST_Attacked:
            SumV += (i - MinV)
        ESTIMATE_DIST_Attacked = (ESTIMATE_DIST_Attacked - MinV) * 1.0 / SumV
        # print(ESTIMATE_DIST_Attacked)

        ESTIMATE_DIST = ESTIMATE_DIST * 1.0 / (USERNUM)
        MinV = min(ESTIMATE_DIST)
        SumV = 0
        for i in ESTIMATE_DIST:
            SumV += (i - MinV)
        ESTIMATE_DIST = (ESTIMATE_DIST - MinV) * 1.0 / SumV
        # print(ESTIMATE_DIST)
        Gain = 0
        for i in Targets:
            Gain += ESTIMATE_DIST_Attacked[i] - ESTIMATE_DIST[i]
        print('After De-Attack(Normalization) Gain is :', Gain)
        '''

        if attackway == RPA or  attackway == MGA:
            DeAttackNewUserNUM, DeAttack_DIST = oue.deattack(UserRepresentation,p,q,USERNUM+NumofAttacks,domain,Targets,USERNUM)
            ESTIMATE_DIST_Attacked = oue.aggregate(DeAttack_DIST, DeAttackNewUserNUM, p, q)
            ERROR_Attacked = oue.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)

            print('DetectedAbnormalUsers:', (USERNUM+NumofAttacks-DeAttackNewUserNUM))
            print('Estimated Beta:', (USERNUM+NumofAttacks-DeAttackNewUserNUM)/(USERNUM+NumofAttacks))

            print(Targets)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            print('NumofAttacks:',NumofAttacks)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / DeAttackNewUserNUM - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('After De-Attack(Detection) Gain is:', Gain1)


    elif perturbway == KRR:
        g = domain
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / (math.exp(epsilon) + g - 1)
        print('g,p,q:', g,p,q)
        print('perturbway:', perturbway)
        perturbed_original_dataset = krr.perturb(original_dataset, len(original_dataset),  g, p, q)
        print('perturbed_original_dataset[0:20]: ', perturbed_original_dataset[0:20])
        print('UserNum:', USERNUM)
        REAL_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(USERNUM):
            REAL_DIST[original_dataset[i]] += 1
        ESTIMATE_DIST = krr.aggregate(perturbed_original_dataset, g, USERNUM, p,q)
        ERROR = krr.error_metric(REAL_DIST, ESTIMATE_DIST, domain)
        print('REAL_DIST[0:20]:', REAL_DIST[0:20])
        print('ESTIMATE_DIST[0:20]: ', ESTIMATE_DIST[0:20])
        print('ERROR:', ERROR)
        Copy_original_dataset = perturbed_original_dataset.copy()
        '''
        Targets = []
        TargetValues = EnumTargetsCloseToFt(REAL_DIST, NumofTarget, F_t * USERNUM)[0]
        print('TargetValues:', TargetValues)
        for v in TargetValues:
            tmp = 0
            for f in REAL_DIST:
                flag = False
                if f == v:
                    Targets.append(tmp)
                    flag = True
                tmp += 1
                if flag:
                    break
        print('Targets:', Targets)
        '''

        Targets = EnumTargetsCloseToFt(REAL_DIST, NumofTarget, F_t * USERNUM)
        print('Targets:', Targets)

        if len(Targets) != NumofTarget:
            print('Error!!! len(Targets) != NumofTarget')
        print('REAL_DIST[Targets[0]]:', REAL_DIST[Targets[0]])
        NumofAttacks = int(Beta * USERNUM / (1 - Beta))
        print('NumofAttacks:', NumofAttacks)
        if attackway == RPA:
            # generate poisoning data
            for i in range(NumofAttacks):
                Copy_original_dataset = np.append(Copy_original_dataset, np.random.randint(g))

            ESTIMATE_DIST_Attacked = krr.aggregate(Copy_original_dataset, domain, len(Copy_original_dataset), p, q)
            ERROR_Attacked = krr.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i]*1.0/(USERNUM+NumofAttacks) - ESTIMATE_DIST[i]/ USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        elif attackway == RIA:
            # generate poisoning data
            poisoning_data = []
            for i in range(NumofAttacks):
                poisoning_data.append(Targets[np.random.randint(len(Targets))])
            perturbed_poisoning_data = krr.perturb(poisoning_data, len(poisoning_data), g, p, q)
            for i in range(NumofAttacks):
                Copy_original_dataset = np.append(Copy_original_dataset, perturbed_poisoning_data[i])

            ESTIMATE_DIST_Attacked = krr.aggregate(Copy_original_dataset, domain, len(Copy_original_dataset),  p, q)
            ERROR_Attacked = krr.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #    FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)

        elif attackway == MGA:
            # generate poisoning data
            poisoning_data = []
            for i in range(NumofAttacks):
                poisoning_data.append(Targets[np.random.randint(len(Targets))])
            perturbed_poisoning_data = poisoning_data
            for i in range(NumofAttacks):
                Copy_original_dataset = np.append(Copy_original_dataset, perturbed_poisoning_data[i])
            ESTIMATE_DIST_Attacked = krr.aggregate(Copy_original_dataset, g, len(Copy_original_dataset), p, q)
            ERROR_Attacked = krr.error_metric(REAL_DIST, ESTIMATE_DIST_Attacked, domain)
            print('REAL_DIST[0:20]:', REAL_DIST[0:20])
            print('ESTIMATE_DIST_Attacked[0:20]: ', ESTIMATE_DIST_Attacked[0:20])
            print('ERROR_Attacked:', ERROR_Attacked)
            Gain1 = 0
            for i in Targets:
                 Gain1 += ESTIMATE_DIST_Attacked[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 = Gain1 * 1.0
            print('Gain is :', Gain1)

            #FT = 0
            #for i in Targets:
            #     FT += REAL_DIST[i] / USERNUM
            #Gain2 = (Gain1 + FT) * 1.0 / FT
            #print('Normalized Gain is :', Gain2)
            Normalize(ESTIMATE_DIST_Attacked.copy(), USERNUM + NumofAttacks, ESTIMATE_DIST.copy(), Targets, USERNUM)



    # ESTIMATE_DIST_Attacked = ESTIMATE_DIST_Attacked*1.0 / (USERNUM+NumofAttacks)
    # MinV = min(ESTIMATE_DIST_Attacked)
    # SumV = 0
    # for i in ESTIMATE_DIST_Attacked:
    #     SumV += (i-MinV)
    # ESTIMATE_DIST_Attacked = (ESTIMATE_DIST_Attacked - MinV) * 1.0 / SumV
    # print(ESTIMATE_DIST_Attacked)
    # Gain = 0
    # for i in Targets:
    #     Gain +=  ESTIMATE_DIST_Attacked[i] - ESTIMATE_DIST[i]/USERNUM
    # print('After De-Attack(Normalization) Gain is :', Gain)

    #ESTIMATE_DIST_Attacked = ESTIMATE_DIST_Attacked * 1.0 / (USERNUM + NumofAttacks)
    ESTIMATE_DIST_Attacked = ESTIMATE_DIST_Attacked * 1.0 / NewUserNUM
    MinV = min(ESTIMATE_DIST_Attacked)
    SumV = 0
    for i in ESTIMATE_DIST_Attacked:
        SumV += (i - MinV)
    ESTIMATE_DIST_Attacked = (ESTIMATE_DIST_Attacked - MinV) * 1.0 / SumV
    #print(ESTIMATE_DIST_Attacked)

    ESTIMATE_DIST = ESTIMATE_DIST * 1.0 / (USERNUM)
    MinV = min(ESTIMATE_DIST)
    SumV = 0
    for i in ESTIMATE_DIST:
        SumV += (i - MinV)
    ESTIMATE_DIST = (ESTIMATE_DIST - MinV) * 1.0 / SumV
    #print(ESTIMATE_DIST)
    Gain = 0
    for i in Targets:
        Gain += ESTIMATE_DIST_Attacked[i] - ESTIMATE_DIST[i]
    print('After De-Attack(Both) Gain is :', Gain)

def Normalize(ESTIMATE_DIST_Attacked, NewUserNUM, ESTIMATE_DIST, Targets,USERNUM):
    ESTIMATE_DIST_Attacked = ESTIMATE_DIST_Attacked * 1.0 / NewUserNUM
    MinV = min(ESTIMATE_DIST_Attacked)
    SumV = 0
    for i in ESTIMATE_DIST_Attacked:
        SumV += (i - MinV)
    ESTIMATE_DIST_Attacked = (ESTIMATE_DIST_Attacked - MinV) * 1.0 / SumV
    # print(ESTIMATE_DIST_Attacked)

    ESTIMATE_DIST = ESTIMATE_DIST * 1.0 / (USERNUM)
    MinV = min(ESTIMATE_DIST)
    SumV = 0
    for i in ESTIMATE_DIST:
        SumV += (i - MinV)
    ESTIMATE_DIST = (ESTIMATE_DIST - MinV) * 1.0 / SumV
    # print(ESTIMATE_DIST)
    Gain = 0
    for i in Targets:
        Gain += ESTIMATE_DIST_Attacked[i] - ESTIMATE_DIST[i]
    print('After Normalization, Gain is :', Gain)

if __name__ == '__main__':
    # distribution = 'uniform'
    #distribution = 'zipf'
    # distribution = 'Fire'
    distribution = 'IPUMS'
    original_dataset = originaldata_generation(distribution)
    # global USERNUM
    # global domain
    USERNUM = len(original_dataset)
    domain = max(original_dataset) + 1
    print('domain: ', domain)

    #perturbway = 'OLH'
    #attackway = 'RPA'

    #perturbway = 'OUE'
    #attackway = 'MGA'

    perturbway = 'OUE'
    attackway = 'RPA'


    deattackway = 'ALL'

    run(perturbway, attackway, original_dataset, epsilon,deattackway)
    print('Data: ', distribution)
    print('epsilon: ', epsilon)
    print('USERNUM: ', USERNUM)
    print('domain: ', domain)
    print('original_dataset[0:20]: ', original_dataset[0:20])
