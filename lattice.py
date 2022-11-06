import numpy as np
import itertools


class Lattice:
    def __init__(self, loe=None):
        self.loe = loe
        self.adj = None
        self.size = np.size(self.loe[0])
        self.majority_matrix = None
        self.minority_matrix = None
        self.join_matrix = None
        self.meet_matrix = None
        self.is_a_lattice = False
        self.dist = False
        self.mod = False

        #compute the matrices of majorities and minorities, for al n,m maj[n,m] = [0,..,0,1,0...] 1 for elements that are >= n,m 0 otherwise
        self.majority_matrix, self.minority_matrix = self.compute_majmin_matrices()
        #compute matrices of join and meet where join_matrix[n,m] = join(n,m)
        self.join_mat, self.meet_mat, self.is_a_lattice = self.compute_joinmeet()

        if self.is_a_lattice:
            self.dist = self.is_distributive()
            self.mod = self.is_modular()
            self.adj = self.loe2adj()

    def is_distributive(self):
        for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            if self.meet_mat[x, self.join_mat[y, z]] != self.join_mat[self.meet_mat[x, y], self.meet_mat[x, z]]:
                return False
        return True

    def is_modular(self):
        for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            if self.join_mat[self.meet_mat[x, y], self.meet_mat[z, y]] != self.meet_mat[self.join_mat[self.meet_mat[x, y], z], y]:
                return False
        return True

    def compute_majmin_matrices(self):
        '''
            Compute the majority/minority matrix multipling raws/columns of loe
        '''
        majority_matrix = np.array(
            [[np.multiply(self.loe[n, :], self.loe[m, :]) for n in range(self.size)] for m in range(self.size)])
        minority_matrix = np.array(
            [[np.multiply(self.loe[:, n], self.loe[:, m]) for n in range(self.size)] for m in range(self.size)])

        return majority_matrix, minority_matrix

    def compute_joinmeet(self):
        '''
            Compute the join/meet matrix multipling loe to majority/minority element rows x columuns and then component-wise
            still to majority/minority element. With these 2 operation a vector which counts the # of majorities grater
            than a majority n is computed and then the join of a pair (a,b) is detached as the majority that as # of majorities
            grater that it = to the corresponding component in the obtained vector.
        '''
        try:
            join_martrix = np.array([[np.multiply(np.matmul(self.loe, self.majority_matrix[n, m]),
                                      self.majority_matrix[n, m]).tolist().index(np.matmul(self.majority_matrix[n, m],
                                                                                           self.majority_matrix[n, m]))
                                      for n in range(self.size)] for m in range(self.size)])
            meet_matrix = np.array([[np.multiply(np.matmul(self.minority_matrix[n, m], self.loe),
                                     self.minority_matrix[n, m]).tolist().index(np.matmul(self.minority_matrix[n, m],
                                                                                          self.minority_matrix[n, m]))
                                     for n in range(self.size)] for m in range(self.size)])
            is_a_lattice = True
        except:
            join_martrix = None
            meet_matrix = None
            is_a_lattice = False

        return join_martrix, meet_matrix, is_a_lattice

    def loe2adj(self, reflexive=False):
        adj = np.copy(self.loe)
        if not reflexive:
            for i in range(self.size):
                adj[i, i] = 0

        for i in range(self.size):
            for j in range(i + 1, self.size):
                if adj[i, j] == 1:
                    for k in range(j + 1, self.size):
                        if adj[j, k] == 1:
                            adj[i, k] = 0
        return adj

