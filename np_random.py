import multiprocessing as mp

import numpy as np

def gen_random_legacy(dummy):
    return np.random.uniform(0., 1., (2,))

def gen_random_new(dummy):
    rng = np.random.default_rng()
    return rng.uniform(0., 1., (2,))

def main():
    pool = mp.Pool(processes=5)
    args = [None] * 5
    results = pool.map(gen_random_legacy, args)
    pool.close()
    pool.join()
    
    for idx, sub_res in enumerate(results):
        print(f'process {idx}: {sub_res}')
    pass

if __name__ == "__main__":
    main()