import math
import numpy as np
def perturb(X, n, g, p, q):
    Y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        v = X[i]
        y = v
        p_sample = np.random.random_sample()
        if p_sample > p - q:
            # perturb
            y = np.random.randint(0, g)
        Y[i] = y
    return Y

def aggregate(Y,domain,n,p,q):
    ESTIMATE_DIST = np.zeros(domain,dtype=np.int32)
    for i in range(n):
        ESTIMATE_DIST[Y[i]] += 1
    #a = 1.0 / n
    #b = q
    a = 1.0
    b = q * n
    ESTIMATE_DIST = (a * ESTIMATE_DIST - b) / (p-q)
    return ESTIMATE_DIST


def error_metric(REAL_DIST,ESTIMATE_DIST,domain):
    abs_error = 0.0
    for x in range(domain):
        # print REAL_DIST[x], ESTIMATE_DIST[x]
        abs_error += np.abs(REAL_DIST[x] - ESTIMATE_DIST[x]) ** 2
    return abs_error / domain