from __future__ import division
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

def get_neighbor(site, L=32):
    pbc = lambda s, d, l: ((s+d)%l + l) % l
    x, y = int(site%L), int(site/L)
    neighbors = []
    xp = pbc(x, +1, L)
    xm = pbc(x, -1, L)
    yp = pbc(y, +1, L)
    ym = pbc(y, -1, L)
    neighbors.append(xp + y  * L)
    neighbors.append(x  + ym * L)
    neighbors.append(xm + y  * L)
    neighbors.append(x  + yp * L)
    return neighbors

def cal_energy(state, L=32):
    eng = 0.0
    J = 1.0
    for site, spin in enumerate(state):
        neighbors = get_neighbor(site, L)
        se = np.sum(state[neighbors], dtype=np.float32)
        eng += J * spin * se
    eng = eng / 1024.0
    return eng
