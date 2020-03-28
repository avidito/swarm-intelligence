import numpy as np
import matplotlib.pyplot as plt
import seaborn

import swarm_intelligence.toolbox as tb

class FireflyAlgorithm():

    ############################################ BUILT-IN ##############################################
    # Class Initiator
    def __init__(self, function, size=100, ndim=2, min_value=-10.0, max_value=10.0, iteration=20,
                 task="minimize", alpha=0.25, gamma=0.9):
        self._size = size
        self._ndim = ndim
        self._min_value = min_value
        self._max_value = max_value
        self._iteration = iteration
        self._function = function
        self._task = task
        self._alpha = alpha
        self._gamma = gamma

        self._init_firefly()
        self._objective = tb.calculate_objective(self._firefly, self._function)
        self._brightness = tb.calculate_fitness(self._objective, self._task)
        self._fbest = self._find_fbest()

    # Class Representative
    def __repr__(self):
        print("""Firefly Algorithm\n
                 Colony Size : {}\n
                 Dimension   : {}\n
                 Iteration   : {}\n
                 Alpha       : {}\n
                 Gamma       : {}""".format(self._size, self._ndim, self._iteration, self._alpha,
                                            self._gamma))

    ############################################# PRIVATE ##############################################
    # Inisiasi Kunang-Kunang
    def _init_firefly(self):
        colony = tb.interval_random(self._size, self._ndim, self._min_value, self._max_value)
        self._firefly = np.array(colony, np.float32)

    # Menentukan Fbest
    def _find_fbest(self):
        fbest_val = -1
        fbest_idx = -1

        for i in range(self._size):
            if fbest_val < self._brightness[i]:
                fbest_idx = i
                fbest_val = self._brightness[i]
        return fbest_idx

    # Perhitungan Relative Brightness
    def _relative_brightness(self, source, target):
        distance = tb.euclidean_distance(self._firefly[source], self._firefly[target])
        rb = self._brightness[source]/(1 + self._gamma*distance*distance)
        return rb

    # Gerakan Acak
    def _random_move(self):
        return self._alpha*(np.random.random(self._ndim) - 0.5)

    # Merubah Posisi Kunang Kunang
    def _update_position(self, cf, target):
        next_pos = self._firefly[cf] \
                 + self._relative_brightness(target, cf)*(self._firefly[target] - self._firefly[cf]) \
                 + self._random_move()
        vvc = np.vectorize(tb.value_clip)
        next_pos = vvc(next_pos, self._min_value, self._max_value)
        self._firefly[cf] = next_pos

    # Memperbaharui Brightness
    def _update_brightness(self, cf):
        self._objective[cf] = tb.calculate_objective(self._firefly[cf].reshape(1,-1), self._function)
        self._brightness[cf] = tb.calculate_fitness(self._objective[cf], self._task)

    ############################################# PUBLIC ###############################################\
    # Fungsi Perhitungan
    def execute(self, show=False, pause=0.5):
        if show == True:
            plt.style.use('seaborn')
            pt = self._firefly.transpose()
            p = plt.scatter(pt[0], pt[1])
            plt.xlim(self._min_value, self._max_value)
            plt.ylim(self._min_value, self._max_value)
            plt.pause(pause)

        for i in range(self._iteration):
            for f in range(self._size):
                for o in range(self._size):
                    if self._brightness[f] < self._brightness[o]:
                        self._update_position(f,o)
                        self._update_brightness(f)
            fb = self._find_fbest()
            self._firefly[fb] += self._random_move()
            self._update_brightness(fb)
            self._fbest = fb

            if show == True:
                pt = self._firefly.transpose()
                p = plt.scatter(pt[0], pt[1])
                plt.xlim(self._min_value, self._max_value)
                plt.ylim(self._min_value, self._max_value)
                plt.pause(pause)
