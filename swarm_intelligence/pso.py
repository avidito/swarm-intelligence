# Particle Swarm Optimization
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import utility.si_utility as usi

class ParticleSwarmOptimization():

    ############################################ BUILT-IN ##############################################
    # Class Initiator
    def __init__(self, function, size=100, ndim=2, min_val=-100.0, max_val=100.0, iteration=100,
                 objective=usi.minimize, target=None, inertia=1.0, c1=2, c2=2, error_lim=1e-5):
        self._size = size
        self._ndim = ndim
        self._particle = self._init_particle(min_val, max_val)
        self._iteration = iteration
        self._function = function
        self._objective = objective
        self._target = target
        self._pbest = np.zeros((size, ndim), np.float32)
        self._pbest_val = np.zeros((size), np.float32)
        self._gbest = np.zeros((ndim), np.float32)
        self._gbest_val = 0.0

        interval = abs(max_val - min_val)
        self._velocity = usi.interval_random(size, ndim, -(interval/10), interval/10)
        self._min = min_val
        self._max = max_val
        self._inertia = inertia
        self._c1 = c1
        self._c2 = c2
        self._error_lim = error_lim
        self._error = None

    # Class Representative
    def __repr__(self):
        print("""Standard Particle Swarm Optimization (PSO)\n
                 Population  : {}\n
                 Dimension   : {}\n
                 Iteration   : {}\n
                 Error Limit : {}\n""".format(self._size, self._ndim, self._iteration, self._error))

    ############################################# PRIVATE ##############################################
    # Fungsi Inisiasi Partikel
    def _init_particle(self, min_val, max_val):
        val = usi.interval_random(self._size, self._ndim, min_val, max_val)
        particle = np.array(val, np.float32)
        return particle

    # Fungsi Menghitung Fitness Score
    def _calculate_fitness(self):
        fitness = np.zeros((self._size), np.float32)

        for i in range(self._size):
            fitness[i] = round(self._function(self._particle[i]), 10)
        return fitness

    # Fungsi Memperbaharui Personal Best
    def _update_pbest(self):
        tfit = np.copy(self._fitness)

        if self._target != None:
            error = 0.0
            for i in range(self._size):
                tfit[i] = abs(self._target - tfit[i])
                erorr += tfit[i]**2
            self._error = sqrt(error)

        vround = np.vectorize(round)
        tfit = vround(self._objective(tfit), 10)
        for i in range(self._size):
            if self._pbest_val[i] < tfit[i]:
                self._pbest[i] = self._particle[i]
                self._pbest_val[i] = tfit[i]

    # Fungsi Memperbaharui Global Best
    def _update_gbest(self):
        for i in range(self._size):
            if self._gbest_val < self._pbest_val[i]:
                self._gbest = self._pbest[i]
                self._gbest_val = self._pbest_val[i]

    # FUngsi Menghitung Kecepatan Partikel
    def _update_velocity(self, c1, c2):
        self._velocity = np.zeros((self._size, self._ndim), np.float32)

        for i in range(self._size):
            for j in range(self._ndim):
                r1 = np.random.random()
                r2 = np.random.random()
                self._velocity[i,j] = (self._inertia*self._velocity[i,j]
                                     + c1*r1*(self._pbest[i,j] - self._particle[i,j])
                                     + c2*r2*(self._gbest[j] - self._particle[i,j]))

    ############################################# PUBLIC ###############################################
    # Fungsi Perhitungan
    def execute(self, show=False):
        self._fitness = self._calculate_fitness()
        self._update_pbest()
        self._update_gbest()
        vclip = np.vectorize(usi.value_clip)

        if show == True:
            plt.style.use('seaborn')
            pt = self._particle.transpose()
            p = plt.scatter(pt[0], pt[1])
            plt.xlim(self._min, self._max)
            plt.ylim(self._min, self._max)
            plt.pause(1)

        c1 = self._c1
        c2 = self._c2
        for i in range(self._iteration):
            if self._target != None:
                if self._error < self._error_lim:
                    break

            self._update_velocity(c1, c2)
            if (i+1)%50 == 0:
                c1 /= 2
                c2 /= 2
            self._particle += self._velocity
            self._particle = vclip(self._particle, self._min, self._max)
            self._fitness = self._calculate_fitness()
            self._update_pbest()
            self._update_gbest()

            if show == True:
                pt = self._particle.transpose()
                plt.scatter(pt[0], pt[1])
                plt.xlim(self._min, self._max)
                plt.ylim(self._min, self._max)
                plt.pause(0.5)
