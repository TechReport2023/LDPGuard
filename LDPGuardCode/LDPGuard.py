from itertools import combinations
import numpy as np
import OLH as olh
import OUE as oue
import KRR as krr
import math
import xxhash
from scipy.special import comb

ZIPF = 'zipf'
UNIFORM = 'uniform'
FIRE = 'Fire'
IPUMS = 'IPUMS'
#USERNUM = 1000000
#domain = 1024
#domain = 100
#USERNUM = 100000
#domain = 100

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
NumofTarget = 10
F_t = 0.01
epsilon = 2.0
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


def array_compare(a_perturbed_original_dataset1, a_perturbed_original_dataset2, comparedbits):
    returnv = True
    for item in comparedbits:
        if item not in a_perturbed_original_dataset1 and   item in a_perturbed_original_dataset2:
            return False
        if item in a_perturbed_original_dataset1 and   item not in a_perturbed_original_dataset2:
            return False
    return returnv



def LDPGuard(perturbway, attackway, original_dataset, epsilon,deattackway):
    global NumofAttacks
    # ESTIMATE_DIST_Attacked = []
    # ESTIMATE_DIST = []
    # Targets = []
    UserRepresentation = []
    #NewUserNUM = USERNUM + NumofAttacks
    if perturbway == OLH:
        g = int(round(math.exp(float(epsilon)))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / (math.exp(epsilon) + g - 1)

        epsilon1 = epsilon * 1.0 / 2
        epsilon2 = epsilon - epsilon1

        p1 = math.exp(epsilon1) / (math.exp(epsilon1) + g - 1)
        q1 = 1.0 / (math.exp(epsilon1) + g - 1)

        p2 = math.exp(epsilon2) / (math.exp(epsilon2) + g - 1)
        q2 = 1.0 / (math.exp(epsilon2) + g - 1)

        print('g,p1,q1:', g, p1, q1)
        print('g,p2,q2:', g, p2, q2)
        print('Perturb way:', perturbway)
        print('UserNum:', USERNUM)

        REAL_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(USERNUM):
            REAL_DIST[original_dataset[i]] += 1
        print('REAL_DIST[0:20]: ', REAL_DIST[0:20])

        seeds = np.random.randint(0, len(original_dataset), len(original_dataset))
        perturbed_original_dataset = olh.perturbWithSeeds(original_dataset, len(original_dataset), g, p, q, seeds)
        print('perturbed_original_dataset[0:20]: ', perturbed_original_dataset[0:20])
        ESTIMATE_DIST = olh.aggregateWithSeeds(perturbed_original_dataset, domain, USERNUM, g, p, q, seeds)
        print('ESTIMATE_DIST[0:20]: ', ESTIMATE_DIST[0:20])

        perturbed_original_dataset1 = olh.perturbWithSeeds(original_dataset, len(original_dataset), g, p1, q1, seeds)
        print('perturbed_original_dataset1[0:20]: ', perturbed_original_dataset1[0:20])
        perturbed_original_dataset2 = olh.perturbWithSeeds(original_dataset, len(original_dataset), g, p2, q2, seeds)
        print('perturbed_original_dataset2[0:20]: ', perturbed_original_dataset2[0:20])
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

        Copy_original_dataset1 = perturbed_original_dataset1.copy()
        Copy_original_dataset2 = perturbed_original_dataset2.copy()

        DATASET1 = []
        DATASET2 = []
        UPDATESEEDS1 = []
        UPDATESEEDS2 = []

        randomsize = 0

        if attackway == RPA:
            # 1. generate poisoning data
            poisoning_data_seeds = np.random.randint(0, domain, NumofAttacks)

            poisoning_data1 = np.random.randint(0, domain, NumofAttacks)
            poisoning_data2 = np.random.randint(0, domain, NumofAttacks)
            encoded_poisoning_data1 = olh.EncodewithSelectedSeeds(poisoning_data1, len(poisoning_data1), g, p1, q1,poisoning_data_seeds)
            encoded_poisoning_data2 = olh.EncodewithSelectedSeeds(poisoning_data2, len(poisoning_data2), g, p2, q2,poisoning_data_seeds)

            count = 0
            for i in range(NumofAttacks):
                item1 = encoded_poisoning_data1[i]
                item2 = encoded_poisoning_data2[i]
                Copy_original_dataset1 = np.append(Copy_original_dataset1, item1)
                Copy_original_dataset2 = np.append(Copy_original_dataset2, item2)
                if item1 == item2:
                    count += 1
            print('count:', count)

            newseeds = []
            for i in range(len(Copy_original_dataset1)):
                if i < len(original_dataset):
                    newseeds.append(seeds[i])
                else:
                    newseeds.append(poisoning_data_seeds[i - len(original_dataset)])

            # 2. calculate P1 and P2
            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                if perturbed_original_dataset1[i] == perturbed_original_dataset2[i]:
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i < len(perturbed_original_dataset1): pass
                if i >= len(perturbed_original_dataset1) and Copy_original_dataset1[i] == Copy_original_dataset2[i]:
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))
            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            P1 = p1 * p2 + (g - 1) * q1 * q2

            P2 = 1.0 / g

            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            DataCopy_list1 = []
            DataCopy_list2 = []
            for i in range(domain):
                # items = []
                # DataCopy_list1.append(items)
                # DataCopy_list2.append(items)
                DataCopy_list1.append([])
                DataCopy_list2.append([])

            for i in range(len(Copy_original_dataset1)):
                item = Copy_original_dataset1[i]
                DataCopy_list1[item].append(i)

            for i in range(len(Copy_original_dataset2)):
                item = Copy_original_dataset2[i]
                DataCopy_list2[item].append(i)
            len1 = 0
            len2 = 0
            for i in range(len(DataCopy_list1)):
                len1 += len(DataCopy_list1[i])
            for i in range(len(DataCopy_list2)):
                len2 += len(DataCopy_list2[i])
            print(len1)
            print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):
                # print('i:',i)
                # item = np.random.randint(len(Targets))
                #item = Targets[np.random.randint(len(Targets))]
                item = np.random.randint(g)

                if len(DataCopy_list1[item]) > 0:
                    DataCopy_list1[item].remove(DataCopy_list1[item][0])
                if len(DataCopy_list2[item]) > 0:
                    DataCopy_list2[item].remove(DataCopy_list2[item][0])

            for items in DataCopy_list1:
                for item in items:
                    DATASET1.append(Copy_original_dataset1[item])
                    UPDATESEEDS1.append(newseeds[item])
            for items in DataCopy_list2:
                for item in items:
                    DATASET2.append(Copy_original_dataset2[item])
                    UPDATESEEDS2.append(newseeds[item])
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))
        elif attackway == RIA:
            # 1. generate poisoning data
            poisoning_data_seeds = np.random.randint(0, domain, NumofAttacks)

            poisoning_data1 = []
            poisoning_data2 = []
            for i in range(NumofAttacks):
                poisoning_data1.append(Targets[np.random.randint(len(Targets))])
                poisoning_data2.append(Targets[np.random.randint(len(Targets))])

            encoded_poisoning_data1 = olh.perturbWithSeeds(poisoning_data1, len(poisoning_data1), g, p1, q1,
                                                                  poisoning_data_seeds)
            encoded_poisoning_data2 = olh.perturbWithSeeds(poisoning_data2, len(poisoning_data2), g, p2, q2,
                                                                  poisoning_data_seeds)

            count = 0
            for i in range(NumofAttacks):
                item1 = encoded_poisoning_data1[i]
                item2 = encoded_poisoning_data2[i]
                Copy_original_dataset1 = np.append(Copy_original_dataset1, item1)
                Copy_original_dataset2 = np.append(Copy_original_dataset2, item2)
                if item1 == item2:
                    count += 1
            print('count:', count)

            newseeds = []
            for i in range(len(Copy_original_dataset1)):
                if i < len(original_dataset):
                    newseeds.append(seeds[i])
                else:
                    newseeds.append(poisoning_data_seeds[i - len(original_dataset)])

            # 2. calculate P1 and P2
            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                if perturbed_original_dataset1[i] == perturbed_original_dataset2[i]:
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i < len(perturbed_original_dataset1): pass
                if i >= len(perturbed_original_dataset1) and Copy_original_dataset1[i] == Copy_original_dataset2[i]:
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))
            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            P1 = p1 * p2 + (g - 1) * q1 * q2

            P2 = (1.0 / g * p1 + (1 - 1.0/g) * q1 ) * (1.0 / g * p2 + (1 - 1.0/g) * q2 )  * g

            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            DataCopy_list1 = []
            DataCopy_list2 = []
            for i in range(domain):
                # items = []
                # DataCopy_list1.append(items)
                # DataCopy_list2.append(items)
                DataCopy_list1.append([])
                DataCopy_list2.append([])

            for i in range(len(Copy_original_dataset1)):
                item = Copy_original_dataset1[i]
                DataCopy_list1[item].append(i)

            for i in range(len(Copy_original_dataset2)):
                item = Copy_original_dataset2[i]
                DataCopy_list2[item].append(i)
            len1 = 0
            len2 = 0
            for i in range(len(DataCopy_list1)):
                len1 += len(DataCopy_list1[i])
            for i in range(len(DataCopy_list2)):
                len2 += len(DataCopy_list2[i])
            print(len1)
            print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)

            poisoning_data_seeds_tmp = np.random.randint(0, domain, randomsize)

            poisoning_data1_tmp = []
            poisoning_data2_tmp = []
            for i in range(randomsize):
                poisoning_data1_tmp.append(Targets[np.random.randint(len(Targets))])
                poisoning_data2_tmp.append(Targets[np.random.randint(len(Targets))])

            encoded_poisoning_data1_tmp = olh.perturbWithSeeds(poisoning_data1_tmp, len(poisoning_data1_tmp), g, p1, q1,
                                                           poisoning_data_seeds_tmp)
            encoded_poisoning_data2_tmp = olh.perturbWithSeeds(poisoning_data2_tmp, len(poisoning_data2_tmp), g, p2, q2,
                                                           poisoning_data_seeds_tmp)

            for i in range(randomsize):
                # print('i:',i)
                # item = np.random.randint(len(Targets))
                # item = Targets[np.random.randint(len(Targets))]
                #item = np.random.randint(g)

                item1 = encoded_poisoning_data1_tmp[i]
                item2 = encoded_poisoning_data2_tmp[i]

                if len(DataCopy_list1[item1]) > 0:
                    DataCopy_list1[item1].remove(DataCopy_list1[item1][0])
                if len(DataCopy_list2[item2]) > 0:
                    DataCopy_list2[item2].remove(DataCopy_list2[item2][0])

            for items in DataCopy_list1:
                for item in items:
                    DATASET1.append(Copy_original_dataset1[item])
                    UPDATESEEDS1.append(newseeds[item])
            for items in DataCopy_list2:
                for item in items:
                    DATASET2.append(Copy_original_dataset2[item])
                    UPDATESEEDS2.append(newseeds[item])
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))
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

            # 1. generate poisoning data
            #poisoning_data_seeds = np.random.randint(0, domain, NumofAttacks)
            #poisoning_data1 = np.random.randint(0, domain, NumofAttacks)
            #poisoning_data2 = np.random.randint(0, domain, NumofAttacks)

            poisoning_data_seeds = []
            poisoning_data1 = []
            poisoning_data2 = []
            for i in range(NumofAttacks):
                poisoning_data1.append(Targets[np.random.randint(len(Targets))])
                poisoning_data2.append(Targets[np.random.randint(len(Targets))])
                poisoning_data_seeds.append(maxseed)

            encoded_poisoning_data1 = olh.EncodewithSelectedSeeds(poisoning_data1, len(poisoning_data1), g, p1, q1,
                                                                  poisoning_data_seeds)
            encoded_poisoning_data2 = olh.EncodewithSelectedSeeds(poisoning_data2, len(poisoning_data2), g, p2, q2,
                                                                  poisoning_data_seeds)

            count1 = 0
            for i in range(NumofAttacks):
                item1 = encoded_poisoning_data1[i]
                item2 = encoded_poisoning_data2[i]
                Copy_original_dataset1 = np.append(Copy_original_dataset1, item1)
                Copy_original_dataset2 = np.append(Copy_original_dataset2, item2)
                if item1 == item2:
                    count1 += 1
            print('count1:', count1)

            newseeds = []
            for i in range(len(Copy_original_dataset1)):
                if i < len(original_dataset):
                    newseeds.append(seeds[i])
                else:
                    newseeds.append(poisoning_data_seeds[i - len(original_dataset)])

            # 2. calculate P1 and P2
            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                if perturbed_original_dataset1[i] == perturbed_original_dataset2[i]:
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i < len(perturbed_original_dataset1): pass
                if i >= len(perturbed_original_dataset1) and Copy_original_dataset1[i] == Copy_original_dataset2[i]:
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))
            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            P1 = p1 * p2 + (g - 1) * q1 * q2

            P2 = (1.0 * maxcount / len(Targets))**2 + (1 - 1.0 * maxcount / len(Targets))**2

            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            DataCopy_list1 = []
            DataCopy_list2 = []
            for i in range(domain):
                # items = []
                # DataCopy_list1.append(items)
                # DataCopy_list2.append(items)
                DataCopy_list1.append([])
                DataCopy_list2.append([])

            for i in range(len(Copy_original_dataset1)):
                item = Copy_original_dataset1[i]
                DataCopy_list1[item].append(i)

            for i in range(len(Copy_original_dataset2)):
                item = Copy_original_dataset2[i]
                DataCopy_list2[item].append(i)
            len1 = 0
            len2 = 0
            for i in range(len(DataCopy_list1)):
                len1 += len(DataCopy_list1[i])
            for i in range(len(DataCopy_list2)):
                len2 += len(DataCopy_list2[i])
            print(len1)
            print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)

            poisoning_data_seeds_tmp = []
            poisoning_data1_tmp = []
            poisoning_data2_tmp = []

            for i in range(randomsize):
                poisoning_data1_tmp.append(Targets[np.random.randint(len(Targets))])
                poisoning_data2_tmp.append(Targets[np.random.randint(len(Targets))])
                poisoning_data_seeds_tmp.append(maxseed)

            encoded_poisoning_data1_tmp = olh.EncodewithSelectedSeeds(poisoning_data1_tmp, len(poisoning_data1_tmp), g, p1, q1,
                                                               poisoning_data_seeds_tmp)
            encoded_poisoning_data2_tmp = olh.EncodewithSelectedSeeds(poisoning_data2_tmp, len(poisoning_data2_tmp), g, p2, q2,
                                                               poisoning_data_seeds_tmp)


            for i in range(randomsize):
                # print('i:',i)
                # item = np.random.randint(len(Targets))
                # item = Targets[np.random.randint(len(Targets))]
                # item = np.random.randint(g)
                item1 = encoded_poisoning_data1_tmp[i]
                item2 = encoded_poisoning_data2_tmp[i]

                if len(DataCopy_list1[item1]) > 0:
                    DataCopy_list1[item1].remove(DataCopy_list1[item1][0])
                if len(DataCopy_list2[item2]) > 0:
                    DataCopy_list2[item2].remove(DataCopy_list2[item2][0])

            for items in DataCopy_list1:
                for item in items:
                    DATASET1.append(Copy_original_dataset1[item])
                    UPDATESEEDS1.append(newseeds[item])
            for items in DataCopy_list2:
                for item in items:
                    DATASET2.append(Copy_original_dataset2[item])
                    UPDATESEEDS2.append(newseeds[item])
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))

        ESTIMATE_DIST_Attacked1 = olh.aggregateWithSeeds(DATASET1, domain, len(DATASET1), g, p1, q1,UPDATESEEDS1)
        ESTIMATE_DIST_Attacked2 = olh.aggregateWithSeeds(DATASET2, domain, len(DATASET2), g, p2, q2,UPDATESEEDS2)
        print('REAL_DIST[0:20]:', REAL_DIST[0:20])
        print('ESTIMATE_DIST_Attacked1[0:20]1: ', ESTIMATE_DIST_Attacked1[0:20])
        print('ESTIMATE_DIST_Attacked1[0:20]2: ', ESTIMATE_DIST_Attacked2[0:20])

        print('USERNUM + NumofAttacks - randomsize:',USERNUM + NumofAttacks - randomsize)


        Gain1 = 0
        for i in Targets:
            #Gain1 += ESTIMATE_DIST_Attacked1[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 += ESTIMATE_DIST_Attacked1[i] * 1.0 / (USERNUM + NumofAttacks - randomsize) - ESTIMATE_DIST[i] / USERNUM
        Gain1 = Gain1 * 1.0
        print('Gain1 is :', Gain1)

        Normalize(ESTIMATE_DIST_Attacked1.copy(), USERNUM + NumofAttacks - randomsize, ESTIMATE_DIST.copy(), Targets,
                  USERNUM)

        Gain2 = 0
        for i in Targets:
            #Gain2 += ESTIMATE_DIST_Attacked2[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain2 += ESTIMATE_DIST_Attacked2[i] * 1.0 / (USERNUM + NumofAttacks - randomsize) - ESTIMATE_DIST[i] / USERNUM
        Gain2 = Gain2 * 1.0
        print('Gain2 is :', Gain2)

        Normalize(ESTIMATE_DIST_Attacked2.copy(), USERNUM + NumofAttacks - randomsize, ESTIMATE_DIST.copy(), Targets,
                  USERNUM)


        return True

    elif perturbway == OUE:
        g = 2 ** domain
        epsilon1 = epsilon * 1.0 / 2
        epsilon2 = epsilon - epsilon1

        p = 1.0 / 2.0
        q = 1.0 / (math.exp(epsilon) + 1)

        p1 = 1.0 / 2.0
        q1 = 1.0 / (math.exp(epsilon1) + 1)
        p2 = 1.0 / 2.0
        q2 = 1.0 / (math.exp(epsilon2) + 1)

        print('g,p1,q1:', g, p1, q1)
        print('g,p2,q2:', g, p2, q2)
        print('Perturb way:', perturbway)
        print('UserNum:', USERNUM)

        REAL_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(USERNUM):
            REAL_DIST[original_dataset[i]] += 1
            uservalue = []
            uservalue.append(original_dataset[i])
            UserRepresentation.append(uservalue)
        print('REAL_DIST[0:20]: ', REAL_DIST[0:20])

        perturbed_original_dataset = oue.perturb(REAL_DIST, USERNUM, p, q)
        print('perturbed_original_dataset[0:20]: ', perturbed_original_dataset[0:20])
        ESTIMATE_DIST = oue.aggregate(perturbed_original_dataset, USERNUM, p, q)
        print('ESTIMATE_DIST[0:20]: ', ESTIMATE_DIST[0:20])
        ##### what it returns is data statistic
        perturbed_original_dataset1 = UserRepresentation.copy()
        POISION_DIST1 = oue.perturb_returnrepresentation(domain, USERNUM, p1, q1, perturbed_original_dataset1)
        print('perturbed_original_dataset1[0:20]: ', perturbed_original_dataset1[0:20])
        perturbed_original_dataset2 = UserRepresentation.copy()
        POISION_DIST2 = oue.perturb_returnrepresentation(domain, USERNUM, p2, q2, perturbed_original_dataset2)
        print('perturbed_original_dataset2[0:20]: ', perturbed_original_dataset2[0:20])
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

        Copy_original_dataset1 = perturbed_original_dataset1.copy()
        Copy_original_dataset2 = perturbed_original_dataset2.copy()

        DATASET1 = []
        DATASET2 = []

        randomsize = 0
        NumOfAdditionalOne = 0

        if attackway == RPA:
            # 1. generate poisoning data
            #count = 0
            for i in range(NumofAttacks):
                noiserepresentation1 = []
                for item1 in range(domain):
                    p_sample1 = np.random.random_sample()
                    if p_sample1 > 0.5:
                        POISION_DIST1[item1] += 1
                        noiserepresentation1.append(item1)
                #Copy_original_dataset1 = np.append(Copy_original_dataset1, noiserepresentation1)
                Copy_original_dataset1.append(noiserepresentation1)
                noiserepresentation2 = []
                for item2 in range(domain):
                    p_sample2 = np.random.random_sample()
                    if p_sample2 > 0.5:
                        POISION_DIST2[item2] += 1
                        noiserepresentation2.append(item2)
                #Copy_original_dataset2 = np.append(Copy_original_dataset2, noiserepresentation2)
                Copy_original_dataset2.append(noiserepresentation2)
                #if np.array_equiv(noiserepresentation1,noiserepresentation2):
                    #count += 1
            #print('count:', count)

            # 2. calculate P1 and P2
            shrinkdomain = domain
            shrinkdomain = 2
            tmparray = np.arange(domain)
            np.random.shuffle(tmparray)
            comparedbits = tmparray[0:shrinkdomain]
            print('comparedbits:',comparedbits)

            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                #if np.array_equiv(perturbed_original_dataset1[i],perturbed_original_dataset2[i]):
                if array_compare(perturbed_original_dataset1[i], perturbed_original_dataset2[i],comparedbits):
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                #if i >= len(perturbed_original_dataset1) and np.array_equiv(Copy_original_dataset1[i],Copy_original_dataset2[i]):
                if i >= len(perturbed_original_dataset1) and array_compare(Copy_original_dataset1[i],Copy_original_dataset2[i],comparedbits):
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))


            # Case1: if real value is selected in comparedbits, then P1 =  P1 = (p1 * p2 + (1 - p1) * (1-p2))*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain - 1)
            # Case2:  if real value is not selected in comparedbits, then P1 =  ( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain)
            # P(Case1) = shrinkdomain * 1.0 / domain
            # P(Case2) = 1 -  P(Case1)
            # Therefore, P1 should be updated by
            # P1 = (shrinkdomain * 1.0 / domain) * (p1 * p2 + (1 - p1) * (1 - p2)) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (shrinkdomain - 1) + (1- shrinkdomain * 1.0 / domain)*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain)

            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            #P1 = (p1 * p2 + (1 - p1) * (1-p2))*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain - 1)
            P1 = (shrinkdomain * 1.0 / domain) * (p1 * p2 + (1 - p1) * (1 - p2)) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (
                        shrinkdomain - 1) + (1 - shrinkdomain * 1.0 / domain) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (
                     shrinkdomain)

            P2 = 1.0 / (2**shrinkdomain)
            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            # DataCopy_list1 = []
            # DataCopy_list2 = []
            # for i in range(domain):
            #     DataCopy_list1.append([])
            #     DataCopy_list2.append([])
            #
            # for i in range(len(Copy_original_dataset1)):
            #     item = Copy_original_dataset1[i]
            #     DataCopy_list1[item].append(i)
            #
            # for i in range(len(Copy_original_dataset2)):
            #     item = Copy_original_dataset2[i]
            #     DataCopy_list2[item].append(i)
            # len1 = 0
            # len2 = 0
            # for i in range(len(DataCopy_list1)):
            #     len1 += len(DataCopy_list1[i])
            # for i in range(len(DataCopy_list2)):
            #     len2 += len(DataCopy_list2[i])
            # print(len1)
            # print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):
                for item1 in range(domain):
                    p_sample1 = np.random.random_sample()
                    if p_sample1 > 0.5 and POISION_DIST1[item1] > 0:
                        POISION_DIST1[item1] -= 1
                for item2 in range(domain):
                    p_sample2 = np.random.random_sample()
                    if p_sample2 > 0.5 and POISION_DIST1[item2] > 0:
                        POISION_DIST2[item2] -= 1
            DATASET1 = POISION_DIST1
            DATASET2 = POISION_DIST2
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))
        elif attackway == RIA:
            # 1. generate poisoning data
            # count = 0
            for i in range(NumofAttacks):
                noiserepresentation1 = []
                item1 = Targets[np.random.randint(len(Targets))]
                noiserepresentation1.append(item1)
                repre1 = []
                repre1.append(noiserepresentation1)
                oue.perturb_returnrepresentation(domain, 1, p1, q1, repre1)
                Copy_original_dataset1.append(repre1[0])

                noiserepresentation2 = []
                item2 = Targets[np.random.randint(len(Targets))]
                noiserepresentation2.append(item2)
                repre2 = []
                repre2.append(noiserepresentation2)
                oue.perturb_returnrepresentation(domain, 1, p2, q2, repre2)
                Copy_original_dataset2.append(repre2[0])

            # 2. calculate P1 and P2
            shrinkdomain = domain
            shrinkdomain = 2
            tmparray = np.arange(domain)
            np.random.shuffle(tmparray)
            comparedbits = tmparray[0:shrinkdomain]
            print('comparedbits:', comparedbits)

            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                # if np.array_equiv(perturbed_original_dataset1[i],perturbed_original_dataset2[i]):
                if array_compare(perturbed_original_dataset1[i], perturbed_original_dataset2[i], comparedbits):
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i >= len(perturbed_original_dataset1) and np.array_equiv(Copy_original_dataset1[i],Copy_original_dataset2[i]):
                if i >= len(perturbed_original_dataset1) and array_compare(Copy_original_dataset1[i],
                                                                           Copy_original_dataset2[i], comparedbits):
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))

            # For P1
            # Case1: if real value is selected in comparedbits, then P1 =  P1 = (p1 * p2 + (1 - p1) * (1-p2))*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain - 1)
            # Case2:  if real value is not selected in comparedbits, then P1 =  ( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain)
            # P(Case1) = shrinkdomain * 1.0 / domain
            # P(Case2) = 1 -  P(Case1)
            # Therefore, P1 should be updated by
            # P1 = (shrinkdomain * 1.0 / domain) * (p1 * p2 + (1 - p1) * (1 - p2)) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (shrinkdomain - 1) + (1- shrinkdomain * 1.0 / domain)*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain)

            #For P2
            # k1 (k1 <= len(Targets)) values in Targets  are selected into comparedbits,
            # shrinkdomain - k1  values must be selected from Non-Targets
            # Therefore, P2 should be updated by
            # P2 = (  (1.0 /len(Targets) * p1 + (1 - 1.0 /len(Targets)) * q1) *
            #         (1.0 /len(Targets) * p2 + (1 - 1.0 /len(Targets)) * q2)      +
            #         (1.0 /len(Targets) * (1 - p1) + (1 - 1.0 /len(Targets)) * (1 - q1) ) *
            #         (1.0 / len(Targets) * (1 - p2) + (1 - 1.0 / len(Targets)) * (1 - q2))
            #      )**k1  *
            #      (q1 * q2 + (1- q1) * (1 - q2) )**(shrinkdomain - k1)

            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            # P1 = (p1 * p2 + (1 - p1) * (1-p2))*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain - 1)
            P1 = (shrinkdomain * 1.0 / domain) * (p1 * p2 + (1 - p1) * (1 - p2)) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (
                    shrinkdomain - 1) + (1 - shrinkdomain * 1.0 / domain) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (
                     shrinkdomain)

            P2 = 0
            for k1 in range(shrinkdomain):
                f  = comb(domain - len(Targets), shrinkdomain - k1) * comb(len(Targets), k1) / comb(domain, shrinkdomain)
                P2 +=  f * (((1.0 / len(Targets) * p1 + (1 - 1.0 / len(Targets)) * q1) *
                     (1.0 /len(Targets) * p2 + (1 - 1.0 / len(Targets)) * q2)+
                     (1.0 /len(Targets) * (1 - p1) + (1 - 1.0 /len(Targets)) * (1 - q1) ) *
                     (1.0 / len(Targets) * (1 - p2) + (1 - 1.0 / len(Targets)) * (1 - q2))
                  )**k1  *  (q1 * q2 + (1- q1) * (1 - q2) )**(shrinkdomain - k1))



            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):

                noiserepresentation1 = []
                item1 = Targets[np.random.randint(len(Targets))]
                noiserepresentation1.append(item1)
                repre1 = []
                repre1.append(noiserepresentation1)
                oue.perturb_returnrepresentation(domain, 1, p1, q1, repre1)
                for item in repre1[0]:
                    POISION_DIST1[item] -= 1

                noiserepresentation2 = []
                item2 = Targets[np.random.randint(len(Targets))]
                noiserepresentation2.append(item2)
                repre2 = []
                repre2.append(noiserepresentation2)
                oue.perturb_returnrepresentation(domain, 1, p2, q2, repre2)
                for item in repre2[0]:
                    POISION_DIST2[item] -= 1

            DATASET1 = POISION_DIST1
            DATASET2 = POISION_DIST2
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))
        elif attackway == MGA:
            # 1. generate poisoning data
            # count = 0
            for i in range(NumofAttacks):
                noiserepresentation1 = []
                for item in Targets:
                    POISION_DIST1[item] += 1
                    noiserepresentation1.append(item)
                NumOfAdditionalOne = int(p + (domain - 1) * q - NumofTarget)
                if NumOfAdditionalOne > 0:
                    TMPIDList1 = []
                    for j in range(domain):
                        if j not in Targets:
                            TMPIDList1.append(j)
                    #print('len(TMPIDList1):', len(TMPIDList1))
                    #print('NumOfAdditionalOne:', NumOfAdditionalOne)
                    SelectedUntargetID1 = np.random.choice(len(TMPIDList1), NumOfAdditionalOne, replace=False)
                    for j in range(NumOfAdditionalOne):
                        item = TMPIDList1[SelectedUntargetID1[j]]
                        POISION_DIST1[item] += 1
                        noiserepresentation1.append(item)
                    Copy_original_dataset1.append(noiserepresentation1)
                else:
                    NumOfAdditionalOne = 0

                noiserepresentation2 = []
                for item in Targets:
                    POISION_DIST2[item] += 1
                    noiserepresentation2.append(item)
                NumOfAdditionalOne = int(p + (domain - 1) * q - NumofTarget)
                if NumOfAdditionalOne > 0:
                    TMPIDList2 = []
                    for j in range(domain):
                        if j not in Targets:
                            TMPIDList2.append(j)
                    SelectedUntargetID2 = np.random.choice(len(TMPIDList2), NumOfAdditionalOne, replace=False)
                    for j in range(NumOfAdditionalOne):
                        item = TMPIDList2[SelectedUntargetID2[j]]
                        POISION_DIST2[item] += 1
                        noiserepresentation2.append(item)
                    Copy_original_dataset2.append(noiserepresentation2)
                else:
                    NumOfAdditionalOne = 0
            # 2. calculate P1 and P2
            shrinkdomain = domain
            shrinkdomain = 2
            tmparray = np.arange(domain)
            np.random.shuffle(tmparray)
            comparedbits = tmparray[0:shrinkdomain]
            print('comparedbits:', comparedbits)

            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                # if np.array_equiv(perturbed_original_dataset1[i],perturbed_original_dataset2[i]):
                if array_compare(perturbed_original_dataset1[i], perturbed_original_dataset2[i], comparedbits):
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i >= len(perturbed_original_dataset1) and np.array_equiv(Copy_original_dataset1[i],Copy_original_dataset2[i]):
                if i >= len(perturbed_original_dataset1) and array_compare(Copy_original_dataset1[i],
                                                                           Copy_original_dataset2[i], comparedbits):
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))

            # Case1: if real value is selected in comparedbits, then P1 =  P1 = (p1 * p2 + (1 - p1) * (1-p2))*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain - 1)
            # Case2:  if real value is not selected in comparedbits, then P1 =  ( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain)
            # P(Case1) = shrinkdomain * 1.0 / domain
            # P(Case2) = 1 -  P(Case1)
            # Therefore, P1 should be updated by
            # P1 = (shrinkdomain * 1.0 / domain) * (p1 * p2 + (1 - p1) * (1 - p2)) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (shrinkdomain - 1) + (1- shrinkdomain * 1.0 / domain)*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain)

            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            # P1 = (p1 * p2 + (1 - p1) * (1-p2))*( q1*q2 + (1-q1)*(1-q2) )**(shrinkdomain - 1)
            P1 = (shrinkdomain * 1.0 / domain) * (p1 * p2 + (1 - p1) * (1 - p2)) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (
                    shrinkdomain - 1) + (1 - shrinkdomain * 1.0 / domain) * (q1 * q2 + (1 - q1) * (1 - q2)) ** (
                     shrinkdomain)

            # P2 = 1.0 / (2 ** shrinkdomain)
            P2 = 0
            for k1 in range(shrinkdomain):
                f = comb(domain - len(Targets), shrinkdomain - k1) * comb(len(Targets), k1) / comb(domain, shrinkdomain)
                #P2 += f * 1.0 / comb(domain - len(Targets), k1)
                #P2 += f * 1.0 / ( comb(domain - len(Targets), NumOfAdditionalOne) **2 + comb(domain - len(Targets), domain - len(Targets)-NumOfAdditionalOne) **2 ) ** (shrinkdomain-k1)
                P2 += f * ( ( NumOfAdditionalOne * 1.0/ (domain - len(Targets))  ) ** 2 +
                         ( (domain - len(Targets)- NumOfAdditionalOne) * 1.0/ (domain - len(Targets)) )** 2 ) ** (shrinkdomain - k1)

            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            # DataCopy_list1 = []
            # DataCopy_list2 = []
            # for i in range(domain):
            #     DataCopy_list1.append([])
            #     DataCopy_list2.append([])
            #
            # for i in range(len(Copy_original_dataset1)):
            #     item = Copy_original_dataset1[i]
            #     DataCopy_list1[item].append(i)
            #
            # for i in range(len(Copy_original_dataset2)):
            #     item = Copy_original_dataset2[i]
            #     DataCopy_list2[item].append(i)
            # len1 = 0
            # len2 = 0
            # for i in range(len(DataCopy_list1)):
            #     len1 += len(DataCopy_list1[i])
            # for i in range(len(DataCopy_list2)):
            #     len2 += len(DataCopy_list2[i])
            # print(len1)
            # print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):
                for item in Targets:
                    POISION_DIST1[item] -= 1
                NumOfAdditionalOne = int(p + (domain - 1) * q - NumofTarget)
                TMPIDList1 = []
                for j in range(domain):
                    if j not in Targets:
                        TMPIDList1.append(j)
                SelectedUntargetID1 = np.random.choice(len(TMPIDList1), NumOfAdditionalOne, replace=False)
                for j in range(NumOfAdditionalOne):
                    item = TMPIDList1[SelectedUntargetID1[j]]
                    POISION_DIST1[item] -= 1

                for item in Targets:
                    POISION_DIST2[item] -= 1
                NumOfAdditionalOne = int(p + (domain - 1) * q - NumofTarget)
                TMPIDList2 = []
                for j in range(domain):
                    if j not in Targets:
                        TMPIDList2.append(j)
                SelectedUntargetID2 = np.random.choice(len(TMPIDList2), NumOfAdditionalOne, replace=False)
                for j in range(NumOfAdditionalOne):
                    item = TMPIDList2[SelectedUntargetID2[j]]
                    POISION_DIST2[item] -= 1

            DATASET1 = POISION_DIST1
            DATASET2 = POISION_DIST2
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))

        print('USERNUM + NumofAttacks - randomsize:',USERNUM + NumofAttacks - randomsize)
        ESTIMATE_DIST_Attacked1 = oue.aggregate(DATASET1, USERNUM + NumofAttacks - randomsize, p1, q1)
        ESTIMATE_DIST_Attacked2 = oue.aggregate(DATASET2, USERNUM + NumofAttacks - randomsize, p2, q2)
        print('REAL_DIST[0:20]:', REAL_DIST[0:20])
        print('ESTIMATE_DIST_Attacked1[0:20]:', ESTIMATE_DIST_Attacked1[0:20])
        print('ESTIMATE_DIST_Attacked2[0:20]: ', ESTIMATE_DIST_Attacked2[0:20])
        Gain1 = 0
        for i in Targets:
            # Gain1 += ESTIMATE_DIST_Attacked1[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 += ESTIMATE_DIST_Attacked1[i] * 1.0 / (USERNUM + NumofAttacks- randomsize) - ESTIMATE_DIST[i] / USERNUM
        Gain1 = Gain1 * 1.0
        print('Gain1 is :', Gain1)

        Normalize(ESTIMATE_DIST_Attacked1.copy(), USERNUM + NumofAttacks- randomsize, ESTIMATE_DIST.copy(), Targets, USERNUM)

        Gain2 = 0
        for i in Targets:
            # Gain2 += ESTIMATE_DIST_Attacked2[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain2 += ESTIMATE_DIST_Attacked2[i] * 1.0 / (USERNUM + NumofAttacks - randomsize) - ESTIMATE_DIST[i] / USERNUM
        Gain2 = Gain2 * 1.0
        print('Gain2 is :', Gain2)

        Normalize(ESTIMATE_DIST_Attacked2.copy(), USERNUM + NumofAttacks - randomsize, ESTIMATE_DIST.copy(), Targets, USERNUM)

    elif perturbway == KRR:
        g = domain
        epsilon1 = epsilon * 1.0 / 2
        epsilon2 = epsilon - epsilon1

        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / (math.exp(epsilon) + g - 1)

        p1 = math.exp(epsilon1) / (math.exp(epsilon1) + g - 1)
        q1 = 1.0 / (math.exp(epsilon1) + g - 1)
        p2 = math.exp(epsilon2) / (math.exp(epsilon2) + g - 1)
        q2 = 1.0 / (math.exp(epsilon2) + g - 1)

        print('g,p1,q1:', g, p1, q1)
        print('g,p2,q2:', g, p2, q2)
        print('Perturb way:', perturbway)
        print('UserNum:', USERNUM)

        REAL_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(USERNUM):
            REAL_DIST[original_dataset[i]] += 1
        print('REAL_DIST[0:20]: ', REAL_DIST[0:20])

        perturbed_original_dataset = krr.perturb(original_dataset, len(original_dataset), g, p, q)
        print('perturbed_original_dataset[0:20]: ', perturbed_original_dataset[0:20])
        ESTIMATE_DIST = krr.aggregate(perturbed_original_dataset, g, USERNUM, p,q)
        print('ESTIMATE_DIST[0:20]: ', ESTIMATE_DIST[0:20])

        perturbed_original_dataset1 = krr.perturb(original_dataset, len(original_dataset), g, p1, q1)
        print('perturbed_original_dataset1[0:20]: ', perturbed_original_dataset1[0:20])
        perturbed_original_dataset2 = krr.perturb(original_dataset, len(original_dataset), g, p2, q2)
        print('perturbed_original_dataset2[0:20]: ', perturbed_original_dataset2[0:20])
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

        Copy_original_dataset1 = perturbed_original_dataset1.copy()
        Copy_original_dataset2 = perturbed_original_dataset2.copy()

        DATASET1 = []
        DATASET2 = []

        randomsize = 0

        if attackway == RPA:
            # 1. generate poisoning data
            count = 0
            for i in range(NumofAttacks):
                item1 = np.random.randint(g)
                item2 = np.random.randint(g)
                # Copy_original_dataset1 = np.append(Copy_original_dataset1, np.random.randint(g))
                # Copy_original_dataset2 = np.append(Copy_original_dataset2, np.random.randint(g))
                Copy_original_dataset1 = np.append(Copy_original_dataset1, item1)
                Copy_original_dataset2 = np.append(Copy_original_dataset2, item2)
                if item1 == item2:
                    count += 1
            print('count:', count)

            # 2. calculate P1 and P2
            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                if perturbed_original_dataset1[i] == perturbed_original_dataset2[i]:
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i < len(perturbed_original_dataset1): pass
                if i >= len(perturbed_original_dataset1) and Copy_original_dataset1[i] == Copy_original_dataset2[i]:
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))
            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)
            P1 = p1 * p2 + (g - 1) * q1 * q2
            P2 = 1.0 / g
            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            if alpha > 0.1:
                print('does not belong to [0.1].')
                return False


            DataCopy_list1 = []
            DataCopy_list2 = []
            for i in range(domain):
                # items = []
                # DataCopy_list1.append(items)
                # DataCopy_list2.append(items)
                DataCopy_list1.append([])
                DataCopy_list2.append([])

            for i in range(len(Copy_original_dataset1)):
                item = Copy_original_dataset1[i]
                DataCopy_list1[item].append(i)

            for i in range(len(Copy_original_dataset2)):
                item = Copy_original_dataset2[i]
                DataCopy_list2[item].append(i)
            len1 = 0
            len2 = 0
            for i in range(len(DataCopy_list1)):
                len1 += len(DataCopy_list1[i])
            for i in range(len(DataCopy_list2)):
                len2 += len(DataCopy_list2[i])
            print(len1)
            print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):
                # print('i:',i)
                # item = np.random.randint(len(Targets))
                #item = Targets[np.random.randint(len(Targets))]
                item = np.random.randint(g)

                if len(DataCopy_list1[item]) > 0:
                    DataCopy_list1[item].remove(DataCopy_list1[item][0])
                if len(DataCopy_list2[item]) > 0:
                    DataCopy_list2[item].remove(DataCopy_list2[item][0])

            for items in DataCopy_list1:
                for item in items:
                    DATASET1.append(Copy_original_dataset1[item])
            for items in DataCopy_list2:
                for item in items:
                    DATASET2.append(Copy_original_dataset2[item])
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))
        elif attackway == RIA:
            # 1. generate poisoning data
            poisoning_data1 = []
            poisoning_data2 = []
            count = 0
            for i in range(NumofAttacks):
                item1 = np.random.randint(len(Targets))
                item2 = np.random.randint(len(Targets))
                poisoning_data1.append(Targets[item1])
                poisoning_data2.append(Targets[item2])
                if item1 == item2:
                    count += 1
            print('count:', count)

            perturbed_poisoning_data1 = krr.perturb(poisoning_data1, len(poisoning_data1), g, p1, q1)
            perturbed_poisoning_data2 = krr.perturb(poisoning_data2, len(poisoning_data2), g, p2, q2)

            for i in range(NumofAttacks):
                Copy_original_dataset1 = np.append(Copy_original_dataset1, perturbed_poisoning_data1[i])
                Copy_original_dataset2 = np.append(Copy_original_dataset2, perturbed_poisoning_data2[i])

            # 2. calculate P1 and P2
            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                if perturbed_original_dataset1[i] == perturbed_original_dataset2[i]:
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i < len(perturbed_original_dataset1): pass
                if i >= len(perturbed_original_dataset1) and Copy_original_dataset1[i] == Copy_original_dataset2[i]:
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))
            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)

            P1 = p1 * p2 + (g - 1) * q1 * q2
            P2 = NumofTarget * (1.0 / NumofTarget * p1 + (1 - 1.0 / NumofTarget) * q1) * ( 1.0 / NumofTarget * p2 + (1 - 1.0 / NumofTarget) * q2) + (g - NumofTarget) * q1 * q2

            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            DataCopy_list1 = []
            DataCopy_list2 = []
            for i in range(domain):
                # items = []
                # DataCopy_list1.append(items)
                # DataCopy_list2.append(items)
                DataCopy_list1.append([])
                DataCopy_list2.append([])

            for i in range(len(Copy_original_dataset1)):
                item = Copy_original_dataset1[i]
                DataCopy_list1[item].append(i)

            for i in range(len(Copy_original_dataset2)):
                item = Copy_original_dataset2[i]
                DataCopy_list2[item].append(i)
            len1 = 0
            len2 = 0
            for i in range(len(DataCopy_list1)):
                len1 += len(DataCopy_list1[i])
            for i in range(len(DataCopy_list2)):
                len2 += len(DataCopy_list2[i])
            print(len1)
            print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):
                # print('i:',i)
                # item = np.random.randint(len(Targets))

                tmp_item1 = Targets[np.random.randint(len(Targets))]
                tmp_item2 = Targets[np.random.randint(len(Targets))]
                X1 = []
                X1.append(tmp_item1)
                X2 = []
                X2.append(tmp_item2)
                item1 = krr.perturb(X1, 1, g, p1, q1)[0]
                item2 = krr.perturb(X2, 1, g, p2, q2)[0]
                if len(DataCopy_list1[item1]) > 0:
                    DataCopy_list1[item1].remove(DataCopy_list1[item1][0])
                if len(DataCopy_list2[item2]) > 0:
                    DataCopy_list2[item2].remove(DataCopy_list2[item2][0])

            for items in DataCopy_list1:
                for item in items:
                    DATASET1.append(Copy_original_dataset1[item])
            for items in DataCopy_list2:
                for item in items:
                    DATASET2.append(Copy_original_dataset2[item])
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))
        elif attackway == MGA:
            # 1. generate poisoning data
            poisoning_data1 = []
            poisoning_data2 = []
            count = 0
            for i in range(NumofAttacks):
                item1 = np.random.randint(len(Targets))
                item2 = np.random.randint(len(Targets))
                poisoning_data1.append(Targets[item1])
                poisoning_data2.append(Targets[item2])
                if item1 == item2:
                    count += 1
            print('count:', count)
            # perturbed_poisoning_data1 = krr.perturb(poisoning_data1, len(poisoning_data1), g, p1, q1)
            # perturbed_poisoning_data2 = krr.perturb(poisoning_data2, len(poisoning_data2), g, p2, q2)
            perturbed_poisoning_data1 = poisoning_data1.copy()
            perturbed_poisoning_data2 = poisoning_data2.copy()
            for i in range(NumofAttacks):
                Copy_original_dataset1 = np.append(Copy_original_dataset1, perturbed_poisoning_data1[i])
                Copy_original_dataset2 = np.append(Copy_original_dataset2, perturbed_poisoning_data2[i])

            # 2. calculate P1 and P2
            C = 0
            C1 = 0
            C2 = 0
            for i in range(len(perturbed_original_dataset1)):
                if perturbed_original_dataset1[i] == perturbed_original_dataset2[i]:
                    C1 += 1
            for i in range(len(Copy_original_dataset1)):
                # if i < len(perturbed_original_dataset1): pass
                if i >= len(perturbed_original_dataset1) and Copy_original_dataset1[i] == Copy_original_dataset2[i]:
                    C2 += 1
            print(len(Copy_original_dataset1) - len(perturbed_original_dataset1))
            C = C1 + C2
            print('C1,C2,C:', C1, C2, C)
            P1 = p1 * p2 + (g - 1) * q1 * q2
            P2 = 1.0 / len(Targets)
            alpha = (P1 - C * 1.0 / (USERNUM + NumofAttacks)) / (P1 - P2)

            print('C * 1.0 / (USERNUM + NumofAttacks):', C * 1.0 / (USERNUM + NumofAttacks))
            print('P1,P2:', P1, P2)
            print('alpha:', alpha)
            print('beta:', Beta)

            if alpha < 0 or alpha > 1:
                print('does not belong to [0-1].')
                return False

            DataCopy_list1 = []
            DataCopy_list2 = []
            for i in range(domain):
                # items = []
                # DataCopy_list1.append(items)
                # DataCopy_list2.append(items)
                DataCopy_list1.append([])
                DataCopy_list2.append([])

            for i in range(len(Copy_original_dataset1)):
                item = Copy_original_dataset1[i]
                DataCopy_list1[item].append(i)

            for i in range(len(Copy_original_dataset2)):
                item = Copy_original_dataset2[i]
                DataCopy_list2[item].append(i)
            len1 = 0
            len2 = 0
            for i in range(len(DataCopy_list1)):
                len1 += len(DataCopy_list1[i])
            for i in range(len(DataCopy_list2)):
                len2 += len(DataCopy_list2[i])
            print(len1)
            print(len2)

            # 3. de-attack
            randomsize = int((USERNUM + NumofAttacks) * alpha)
            print('randomsize:', randomsize)
            for i in range(randomsize):
                # print('i:',i)
                # item = np.random.randint(len(Targets))

                item = Targets[np.random.randint(len(Targets))]

                if len(DataCopy_list1[item]) > 0:
                    DataCopy_list1[item].remove(DataCopy_list1[item][0])
                if len(DataCopy_list2[item]) > 0:
                    DataCopy_list2[item].remove(DataCopy_list2[item][0])

            for items in DataCopy_list1:
                for item in items:
                    DATASET1.append(Copy_original_dataset1[item])
            for items in DataCopy_list2:
                for item in items:
                    DATASET2.append(Copy_original_dataset2[item])
            print('len of dataset1:', len(DATASET1))
            print('len of dataset2:', len(DATASET2))

        ESTIMATE_DIST_Attacked1 = krr.aggregate(DATASET1, domain, len(DATASET1), p1, q1)
        ESTIMATE_DIST_Attacked2 = krr.aggregate(DATASET2, domain, len(DATASET2), p2, q2)
        print('REAL_DIST[0:20]:', REAL_DIST[0:20])
        print('ESTIMATE_DIST_Attacked1[0:20]1: ', ESTIMATE_DIST_Attacked1[0:20])
        print('ESTIMATE_DIST_Attacked1[0:20]2: ', ESTIMATE_DIST_Attacked2[0:20])

        print('USERNUM + NumofAttacks - randomsize:',USERNUM + NumofAttacks - randomsize)


        Gain1 = 0
        for i in Targets:
            #Gain1 += ESTIMATE_DIST_Attacked1[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain1 += ESTIMATE_DIST_Attacked1[i] * 1.0 / (USERNUM + NumofAttacks - randomsize) - ESTIMATE_DIST[i] / USERNUM
        Gain1 = Gain1 * 1.0
        print('Gain1 is :', Gain1)

        Normalize(ESTIMATE_DIST_Attacked1.copy(), USERNUM + NumofAttacks - randomsize, ESTIMATE_DIST.copy(), Targets,
                  USERNUM)

        Gain2 = 0
        for i in Targets:
            #Gain2 += ESTIMATE_DIST_Attacked2[i] * 1.0 / (USERNUM + NumofAttacks) - ESTIMATE_DIST[i] / USERNUM
            Gain2 += ESTIMATE_DIST_Attacked2[i] * 1.0 / (USERNUM + NumofAttacks - randomsize) - ESTIMATE_DIST[i] / USERNUM
        Gain2 = Gain2 * 1.0
        print('Gain2 is :', Gain2)

        Normalize(ESTIMATE_DIST_Attacked2.copy(), USERNUM + NumofAttacks - randomsize, ESTIMATE_DIST.copy(), Targets,
                  USERNUM)

        return True

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
    # distribution = 'zipf'
    # distribution = 'Fire'
    distribution = 'IPUMS'
    original_dataset = originaldata_generation(distribution)
    print('original_dataset[0:20]: ', original_dataset[0:20])
    # global USERNUM
    # global domain
    USERNUM = len(original_dataset)
    domain = max(original_dataset) + 1
    print('domain: ', domain)

    perturbway = 'KRR'
    attackway = 'RPA'

    #perturbway = 'OUE'
    #attackway = 'MGA'


    #perturbway = 'OLH'
    #attackway = 'MGA'


    deattackway = 'ALL'

    #run(perturbway, attackway, original_dataset, epsilon,deattackway)

    iter = 0
    Flag =  LDPGuard(perturbway, attackway, original_dataset, epsilon, deattackway)
    print('iter:', iter)
    print('**************************************************')
    while Flag is False:
        Flag = LDPGuard(perturbway, attackway, original_dataset, epsilon, deattackway)
        iter += 1
        print('iter:', iter)
        print('**************************************************')


    print('Data: ', distribution)
    print('epsilon: ', epsilon)
    print('USERNUM: ', USERNUM)
    print('domain: ', domain)
    print('original_dataset[0:20]: ', original_dataset[0:20])
