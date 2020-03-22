import numpy as np
import matplotlib.pyplot as plt
import seaborn

import artificial_intelligence.toolbox as tb

class GreyWolfOptimization():

    ############################################ BUILT-IN ##############################################
    # Class Initatitor
    def __init__(self, function, size=100, ndim=2, min_value=-100.0, max_value=100.0, iteration=100,
                 task=tb.minimize, target=None, error_lim=1e-5):
        self._function = function
        self._size = size
        self._ndim = 2
        self._min_value = min_value
        self._max_value = max_value
        self._iteration = iteration
        self._task = task
        self._target = target
        self._error_lim = error_lim

        self._init_pack()

    # Class Representative
    def __repr__(self):
        print("""Grey Wolf Optimization\n
                 Pack Size : {}\n
                 Dimension : {}\n
                 Iteration : {}\n
                 Min Value : {}\n
                 Max Value : {}""".format(self._size, self._ndim, self._iteration, self._min_value,
                                          self._max_value))

    ############################################# PRIVATE ##############################################
    # Inisiasi Kawanan
    def _init_pack(self):
        val = tb.interval_random(self._size, self._ndim, self._min_value, self._max_value)
        self._pack = np.array(val, np.float32)
        self._a = 2
        self._vA = 2*self._a*np.random.random((3,self._ndim)) - self._a
        self._vC = 2*np.random.random((3, self._ndim))

    # Memperbaharui Nilai Vektor Koefisien
    def _update_coef_vector(self):
        dec = 2/self._iteration
        self._a -= dec
        self._vA = 2*self._a*np.random.random((3,self._ndim)) - self._a
        self._vC = 2*np.random.random((3, self._ndim))

    # Memperbaharui Alpha, Beta, Gamma
    def _update_pack_hierarchy(self):
        self._fitness = tb.calculate_fitness(self._objective, self._task, self._target)
        alpha = 0
        beta = 0
        omega = 0

        for i in range(self._size):
            if self._fitness[alpha] < self._fitness[i]:
                omega = beta
                beta = alpha
                alpha = i
            elif self._fitness[beta] < self._fitness[i]:
                omega = beta
                beta = i
            elif self._fitness[omega] < self._fitness[i]:
                omega = i
        self._alpha = alpha
        self._beta = beta
        self._omega = omega

    # Memperbaharui Posisi Tiap Serigala
    def _update_position(self):
        awolf = self._pack[self._alpha]
        bwolf = self._pack[self._beta]
        owolf = self._pack[self._omega]

        for i in range(self._size):
            if i == self._alpha or i == self._beta or i == self._omega:
                continue
            da = abs(self._vC[0]*awolf - self._pack[i])
            db = abs(self._vC[1]*bwolf - self._pack[i])
            do = abs(self._vC[2]*owolf - self._pack[i])
            xa = awolf - self._vA[0]*da
            xb = bwolf - self._vA[0]*db
            xo = owolf - self._vA[0]*do
            self._pack[i] = (xa + xb + xo)/3

    ############################################# PUBLIC ###############################################
    # Fungsi Perhitungan
    def execute(self, show=False, pause=0.5):
        self._objective = tb.calculate_objective(self._pack, self._function)
        self._update_pack_hierarchy()

        if show == True:
            plt.style.use('seaborn')
            pt = self._pack.transpose()
            p = plt.scatter(pt[0], pt[1])
            plt.xlim(self._min_value, self._max_value)
            plt.ylim(self._min_value, self._max_value)
            plt.pause(pause)

        for i in range(self._iteration):
            self._update_position()
            self._objective = tb.calculate_objective(self._pack, self._function)
            self._update_coef_vector()
            self._update_pack_hierarchy()

            if show == True:
                pt = self._pack.transpose()
                p = plt.scatter(pt[0], pt[1])
                plt.xlim(self._min_value, self._max_value)
                plt.ylim(self._min_value, self._max_value)
                plt.pause(pause)
