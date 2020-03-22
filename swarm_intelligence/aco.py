import numpy as np
import artificial_intelligence.toolbox as tb

class AntColonyOptimization():

    ############################################ BUILT-IN ##############################################
    # Class Initiator
    def __init__(self, distance,  origin, destination, size=100, iteration=100, pheromone_value=0.0,
                 evaporation=0.3):
        self._size = size
        self._iteration = iteration

        self._distance = distance
        drow, dcol = distance.shape
        self._pheromone_value = pheromone_value
        self._pheromone = np.full((drow,dcol), pheromone_value, np.float32)

        self._origin = origin
        self._destination = destination
        self._evaporation = evaporation

    # Class Representative
    def __repr__(self):
        print("""Ant Colony Optimization (ACO)\n
                 Colony Size      : {}\n
                 Origin           : {}\n
                 Destination      : {}\n
                 Iteration        : {}\n
                 Pheromon (Start) : {}\n
                 Evaporation      : {}\n""".format(self._size, self._origin, self._destination,
                                                   self._iteration, self._pheromon_value,
                                                   self._evaporation))

    ############################################# PRIVATE ##############################################
    # Menghitung Probabilitas Pemilihan Jalur
    def _calculate_prob(self, adj):
        prob = np.zeros((len(adj)), np.float32)
        denom = 0.0

        for i in range(len(adj)):
            s, d = adj[i]
            prob[i] = self._pheromone[s,d] + (1/self._distance[s,d])
            denom += prob[i]
        prob /= denom
        return prob

    # Menelusuri Titik dan Panjang Explorasi Semut
    def _explore_to(self, track,  destination, l=0,):
        prev = None

        while(track[-1] != destination):
            cp = track[-1]
            adj = tb.get_adjacent(cp, self._distance, ex=prev)
            prev = track[-1]
            prob = self._calculate_prob(adj)
            choice = tb.roullete_wheel(prob)
            l += self._distance[adj[choice][0], adj[choice][1]]
            if(adj[choice][0] == cp):
                track.append(adj[choice][1])
            else:
                track.append(adj[choice][0])
        return track, l

    # Memperbaharui Feromon
    def _update_pheromone(self, track_list, p_list):
        row, col = self._pheromone.shape

        for i in range(row):
            for j in range(col):
                self._pheromone[i,j] = (1-self._evaporation)*self._pheromone[i,j]

        for i in range(len(track_list)):
            for j in range(len(track_list[i])-1):
                a = track_list[i][j]
                b = track_list[i][j+1]
                if a < b:
                    self._pheromone[a,b] += p_list[i]
                else:
                    self._pheromone[b,a] += p_list[i]

    # Eksplorasi oleh Semut
    def _ant_exploration(self):
        track_list = []
        p_list = []

        for i in range(self._size):
            track = [self._origin]
            l = 0
            prev = None
            track, l = self._explore_to(track, l, self._destination)
            track, l = self._explore_to(track, l, self._origin)
            p = 1/l
            track_list.append(track)
            p_list.append(p)
        return track_list, p_list

    ############################################# PUBLIC ###############################################
    # Fungsi Perhitungan
    def execute(self):
        for it in range(self._iteration):
            track_list, p_list = self._ant_exploration()
            self._update_pheromone(track_list, p_list)

    # Evaluasi Lintasan
    def evaluate_track(self):
        cp = self._origin
        prev = None
        track = [cp]
        l = 0

        while(cp != self._destination):
            adj = tb.get_adjacent(cp, self._distance, ex=prev)
            prev = cp
            bpher = -1
            nxt = -1
            for i in range(len(adj)):
                if adj[i][0] < adj[i][1]:
                    s = adj[i][0]
                    d = adj[i][1]
                else:
                    s = adj[i][1]
                    d = adj[i][0]
                if bpher < self._pheromone[s,d]:
                    bpher = self._pheromone[s,d]
                    if s == cp:
                        nxt = d
                    else:
                        nxt = s
            cp = nxt
            track.append(cp)
            l += self._distance[s,d]
        return track, l
