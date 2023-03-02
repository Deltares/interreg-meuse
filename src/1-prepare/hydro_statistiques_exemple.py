import pandas as pd
import os
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt


# functions copied from Hydromt branch!
def legendre_shift_poly(n):
    """Shifted Legendre polynomial
    Based on recurrence relation
        (n + 1)Pn+1 (x) - (1 + 2 n)(2 x - 1)Pn (x) + n Pn-1 (x) = 0
    Given nonnegative integer n, compute the shifted Legendre polynomial P_n.
    Return the result as a vector whose mth element is the coefficient of x^(n+1-m).
    polyval(legendre_shift_poly(n),x) evaluates P_n(x).
    """

    if n == 0:
        pk = 1
    elif n == 1:
        pk = [2, -1]
    else:

        pkm2 = np.zeros(n + 1)
        pkm2[-1] = 1
        pkm1 = np.zeros(n + 1)
        pkm1[-1] = -1
        pkm1[-2] = 2

        for k in range(2, n + 1):
            pk = np.zeros(n + 1)

            for e in range(n - k + 1, n + 1):
                pk[e - 1] = (
                        (4 * k - 2) * pkm1[e]
                        + (1 - 2 * k) * pkm1[e - 1]
                        + (1 - k) * pkm2[e - 1]
                )

            pk[-1] = (1 - 2 * k) * pkm1[-1] + (1 - k) * pkm2[-1]
            pk = pk / k

            if k < n:
                pkm2 = pkm1
                pkm1 = pk

    return pk


def get_lmom(x, nmom=4):
    """Compute L-moments for a data series.
    Based on calculation of probability weighted moments and the coefficient
    of the shifted Legendre polynomial.
    lmom by Kobus N. Bekker, 14-09-2004
    Parameters:
    -----------
    x: 1D array of float
        data series
    nmom: int
        number of L-Moments to be computed, by default 4.
    Returns:
    --------
    lmom: 1D array of float
        vector of (nmom) L-moments
    """

    n = len(x)
    xs = np.msort(x)
    bb = np.zeros(nmom - 1)
    ll = np.zeros(nmom - 1)
    b0 = xs.mean(axis=0)

    for r in range(1, nmom):
        Num1 = np.kron(np.ones((r, 1)), np.arange(r + 1, n + 1))
        Num2 = np.kron(np.ones((n - r, 1)), np.arange(1, r + 1)).T
        Num = np.prod(Num1 - Num2, axis=0)

        Den = np.prod(np.kron(np.ones((1, r)), n) - np.arange(1, r + 1))
        bb[r - 1] = (((Num / Den) * xs[r:n]).sum()) / n

    B = np.concatenate([np.array([b0]), bb.T])[::-1]

    for i in range(1, nmom):
        Spc = np.zeros(len(B) - (i + 1))
        Coeff = np.concatenate([Spc, legendre_shift_poly(i)])
        ll[i - 1] = np.sum(Coeff * B)

    lmom = np.concatenate([np.array([b0]), ll.T])

    return lmom


folder_p = r'p:\11208719-interreg\data\hydroportail\a_raw\la_meuse_a_chooz\QJ-X_(CRUCAL)'
file_sample = 'QJ-X-(CRUCAL)_Gumbel_B7200000_01-01-1953_07-02-2023_1_non-glissant_X_pre-valide-et-valide_Echantillon.csv'

df = pd.read_csv(os.path.join(folder_p, file_sample))
# length of sample
n = len(df)
# keep only discharge values and rank columns
df = df.loc[:, ['Valeur (en m³/s)', 'Date']]
# reset index
df.index = range(0, n)

# Ajustement statistique estimee par la methode L-moments
x = df['Valeur (en m³/s)'].to_numpy()
res = get_lmom(x, 2)
lambda1 = res[0]
lambda2 = res[1]

# bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95, random_state=1, method='percentile')

# https://echo2.epfl.ch/e-drologie/chapitres/annexes/AnalFrequ.html#cours
# euler's constant
gamma = 0.5772
b = lambda2 / np.log(2)
a = lambda1 - b * gamma

# basee sur https://www.slideshare.net/HAMZAFARIYOU/hydrologie-msi-ch7-hydrologie-statistique
df.sort_values(['Valeur (en m³/s)'], ascending=[True], inplace=True)
df['r'] = range(1, 1 + n)
# Des simulations ont montré que pour la loi de Gumbel, il est judicieux d'utiliser la distribution empirique de Hazen
df['F'] = (df['r'] - 0.5) / n  # Hazen
df['u'] = -np.log(-np.log(df['F']))

u = np.arange(-1, 6, 1)
F = np.exp(-np.exp(-u))

fig, ax = plt.subplots()
ax.plot(df['u'], df['Valeur (en m³/s)'], 'o')
ax.plot(df['u'], df['u'] * b + a, '-')
ax.set_ylabel('Débit (en m³/s)')

plt.show()

df['T'] = 1 / (1 - df['F'])
df['quantile'] = a + b * df['u']


