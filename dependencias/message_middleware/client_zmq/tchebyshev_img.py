import os.path as pth
import pickle
import sys
import os

import cv2
import numpy as np

# from scipy.special import binom, factorial, poch


def normalize_rgb_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_dn = gray / 255.0
    norm = np.sqrt((gray_dn ** 2).sum())
    gray_norm = gray_dn / norm

    return gray_norm


class ImageMomentGenerator:
    def __init__(self, width, height, base_dir="."):

        self.M = width
        self.N = height

        self.t_row = DiscretePolynomial(self.N, base_dir)
        self.t_col = DiscretePolynomial(self.M, base_dir)

        # Invoke computation of polynomials
        self[0, 0]

    def __getitem__(self, key):

        n, m = key
        phi = np.outer(self.t_col[m, :], self.t_row[n, :])

        return phi

    def compute(self, image, n, m):

        tensor = self[n, m] * image
        return tensor.sum()

    def generate_moments(self, image, max_idx):

        moment_indices = get_moment_indices(self.N, self.M, max_idx)

        if self.N != self.M:
            compute = lambda n, m: self.compute(image, n, m)
        else:
            compute_cache = {}

            def compute(n, m):
                s_key = tuple(sorted((n, m)))
                if s_key not in compute_cache:
                    compute_cache[s_key] = self.compute(image, *s_key)
                    # print("m", end='', file=sys.stderr)
                # else:
                # print("h", end='', file=sys.stderr)

                return compute_cache[s_key]

        moments = {(n, m): compute(n, m) for n, m in moment_indices}

        return moments


def moments_norm(moments):

    return np.linalg.norm([moment for moment in moments.values()], 2)


def focus_measure(moments):

    norm = moments_norm(moments)
    return (1 - norm) / norm


###############################################################################
#  Computation with Python integers and dc.Decimals, for numerical stability  #
###############################################################################
import functools
import operator
import decimal as dc

dc.getcontext().prec = 50


class DiscretePolynomial:
    def __init__(self, lenght, base_dir="."):

        self.N = lenght
        self._computed = False
        self._base_dir = base_dir

    def compute_poly_terms(self):
        N = self.N
        terms = []

        # Generate with recurrence relation
        X = list(range(N))
        terms.append([1] * N)
        terms.append([1 - N + 2 * x for x in X])
        for n in range(2, N):
            new_poly = [
                (
                    (2 * n - 1) * (2 * x - N + 1) * tm
                    - (n - 1) * (N ** 2 - (n - 1) ** 2) * tmm
                )
                / dc.Decimal(n)
                for x, tm, tmm in zip(X, terms[-1], terms[-2])
            ]
            terms.append(new_poly)

            # Normalize and check if computation can be stopped
            if n >= 3:
                n_ = n - 3
                poly = DiscretePolynomial.normalize_poly(terms[n_], n_, N)
                success = (np.sum(poly) < 1e-9) or (n_ == 0)
                if success:
                    terms[n_] = poly
                else:
                    del terms[-4:]
                    break
        if success:
            # Last two polynomials might also be successful
            for n in range(N - 3, N):
                poly = DiscretePolynomial.normalize_poly(terms[n], n, N)
                success = np.sum(poly) < 1e-9
                if success:
                    terms[n] = poly

        self._terms = np.array(terms)
        self._computed = True

    def __getitem__(self, key):
        if not self._computed:
            self.unpickle()
        if not self._computed:
            self.compute_poly_terms()
            self.pickle()

        return self._terms[key]

    @property
    def terms(self):
        terms = 0 if not self._computed else self._terms.shape[0]
        return terms

    def MeasureSqrt(n, N):
        prods = [N + i for i in range(-n, n + 1)]
        prod = functools.reduce(operator.mul, prods) / dc.Decimal(2 * n + 1)
        return prod.sqrt()

    @staticmethod
    def normalize_poly(denorm_poly, n, N, measure = MeasureSqrt):
        norm = measure(n, N)
        dec_norm_poly = [dnp / norm for dnp in denorm_poly]
        norm_poly = [float(p) for p in dec_norm_poly]

        return norm_poly

    def pickle(self):
        with open(self.pickle_filename, "wb") as fobj:
            pickle.dump(self._terms, fobj)
            print("Saved polynomial data to {}".format(self.pickle_filename))

    def unpickle(self):
        if pth.exists(self.pickle_filename):
            with open(self.pickle_filename, "rb") as fobj:
                terms = pickle.load(fobj)
                t, N = terms.shape
                assert N == self.N, "Invalid polynomial pickle file: {}".format(
                    self.pickle_filename
                )
                self._terms = terms
                self._computed = True
                print("Loaded polynomial data from {}".format(self.pickle_filename))

    @property
    def pickle_filename(self):
        file_n = "tchebyshev-{}.pkl".format(self.N)
        # return pth.abspath(pth.join(self._base_dir, file_n))
        return pth.abspath(pth.join(self._base_dir, file_n))

    def get_moment_indices(N, M, max_idx):
        return [(n, m) for n in range(N) for m in range(M) if (n + m) <= max_idx]


"""
Numpy and Scipy-based implementation
class DiscretePolynomialNumpy:

    def __init__(self, lenght):

        self.N = lenght
        self._computed = False

    def compute_poly_terms(self):

        N = self.N
        terms = np.empty((N, N), dtype=np.float128)

        # Generate with recurrence relation
        x = np.r_[0:N]
        terms[0, :] = 1
        terms[1, :] = 1 - N + 2*x
        for n in range(2, N):
            terms[n, :] = ( (2*n - 1)*(2*x - N + 1)*terms[n-1, :] -
                            (n - 1)*(N**2 - (n - 1)**2)*terms[n-2, :] )/n

        # Normalize
        for n in range(1, N):
            terms[n, :] /= MeasureSqrtNumpy(n, N)

        self._terms = terms
        self._computed = True

    def __getitem__(self, *args):

        if not self._computed:
            self.compute_poly_terms()

        return self._terms.__getitem__(*args)


def MeasureSqrtNumpy(n, N, formula='prods'):

    if formula == 'binomial':
        measure = np.sqrt(factorial(2*n) * binom(N + n, 2*n + 1))

    elif formula == 'prods':
        prods = np.sqrt(N + np.r_[-n:(n + 1)])
        prods[0] /= np.sqrt(2*n + 1)
        measure = prods.prod() 

    elif formula == 'poch':
        measure = np.sqrt(poch(N - n, 2*n + 1) / (2*n + 1))

    elif formula == 'lnprods':
        prods = np.log(N + np.r_[-n:(n + 1)])
        log_sq = ( prods.sum() - np.log(2*n + 1) )/2
        measure = np.exp(log_sq)

    return measure
"""


"""
import numpy as np
import tchebyshev_img as tc

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

t_p_row = tc.DiscretePolynomial(50)
X = np.r_[0:50]
X, Y = np.meshgrid(X, X)
phi_2_2 = np.outer(t_p_row[2, :], t_p_row[2, :])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, phi_2_2, rstride=1, cstride=1, cmap=cm.coolwarm)
fig.colorbar(surf)

plt.show()
"""
