import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class Phong:
    def __init__(self, n: int) -> None:
        self.n = n
        self.parallel = True

    def sample_uniform(self, n_samples: int = 10000) -> np.ndarray:
        print(f'generating {n_samples} samples uniformly')
        if self.parallel:
            cos = np.random.rand(n_samples)
            cos2 = np.square(cos)
            return cos2 * ((np.clip((-1 + 2 * cos2), a_min=0, a_max=None) ** 2) ** self.n)
        else:
            assert False
    
    def sample_cos2_weighted(self, n_samples: int = 10000) -> np.ndarray:
        print(f'generating {n_samples} samples by cos2-weighted importance sampling')
        if self.parallel:
            cos = np.power(np.random.rand(n_samples), 1./3.)
            cos2 = np.square(cos)
            return ((np.clip((-1 + 2 * cos2), a_min=0, a_max=None) ** 2) ** self.n)
        else:
            assert False
    
    def radial_distribution(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        sin = np.linspace(0., 1., n_samples)
        cos2 = np.sqrt(1. - sin ** 2)
        L = cos2 * (np.clip(-1. + 2. * cos2, a_min=0., a_max=1.) ** self.n)
        return sin, L

def save_histogram(data: np.ndarray, bins: int = 100, out_path: str = 'output/hist.png') -> None:
    data = data.flatten()
    data = np.sort(data)
    hist, _ = np.histogram(data, bins=bins+1)
    data = data[hist[0]:]
    hist, _ = np.histogram(data, bins=bins, density=True)
    plt.clf()
    plt.plot(np.linspace(0, 1, bins), hist)
    plt.savefig(out_path)

if __name__ == '__main__':
    OUT_DIR = './output'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='power for the Phong specular term')
    parser.add_argument('-s', '--samples', type=int, default=int(1e6), help='number of samples')
    parser.add_argument('-o', '--output', default='hist.png', help='output file name')
    parser.add_argument('-b', '--bins', type=int, default=100, help='number of bins')
    args = parser.parse_args()

    out_path = os.path.join(OUT_DIR, args.output)
    plt.clf()
    for n in range(5, 50, 5):
        # phong = Phong(args.n)
        phong = Phong(n)
        x, y = phong.radial_distribution()
        plt.plot(x, y)
        # save_histogram(phong.sample_cos2_weighted(n_samples=args.samples),
        #             bins=args.bins,
        #             out_path=out_path)
    plt.savefig(out_path)
