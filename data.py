import numpy as np

def cal_mag(spins):
    return np.mean(spins)

def brokenstate(L, dev=0.02):
    N = L * L
    '''
    num_flip = int(N * np.random.uniform(low=flip_ratio_range[0],
                                high=flip_ratio_range[1]))
    '''
    num_flip = int(N * np.abs(np.random.normal(0, dev)))
    label = 0
    if (np.random.rand() > 0.5):
        # up spin
        spins = np.ones(N)
        label = +1
    else:
        # down spin
        spins = -np.ones(N)
        label = -1
    
    sites = np.random.choice(N, num_flip)
    spins[sites] *= -1

    return spins, label
