import numpy as np
import itertools
import torch


class Lattice:
    def __init__(self, loe=torch.zeros(1)):
        self.loe = torch.from_numpy(loe)
        self.loe_transposed = torch.transpose(self.loe, 0, 1)
        self.adj = None
        self.size = self.loe.size(dim=1)
        self.majority_tensor = None
        self.minority_tensor = None
        self.join_tensor = None
        self.meet_tensor = None
        self.is_a_lattice = False
        self.dist = False
        self.mod = False

        #compute the matrices of majorities and minorities, for al n,m maj[n,m] = [0,..,0,1,0...] 1 for elements that are >= n,m 0 otherwise
        self.majority_tensor, self.minority_tensor = self.compute_majmin_tensors()
        #compute matrices of join and meet where join_matrix[n,m] = join(n,m)
        self.join_tensor, self.meet_tensor, self.is_a_lattice = self.compute_joinmeet()

        if self.is_a_lattice:
            self.dist = self.is_distributive()
            self.mod = self.is_modular()
            self.adj = self.loe2adj()

    def is_distributive(self):
        for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            if self.meet_tensor[x, self.join_tensor[y, z]] != self.join_tensor[self.meet_tensor[x, y], self.meet_tensor[x, z]]:
                return False
        return True

    def is_modular(self):
        for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            if self.join_tensor[self.meet_tensor[x, y], self.meet_tensor[z, y]] != self.meet_tensor[self.join_tensor[self.meet_tensor[x, y], z], y]:
                return False
        return True

    def is_meet_semidistributive(self):
        for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            if self.meet_tensor[x, y] == self.meet_tensor[x, z] and self.meet_tensor[x, self.join_tensor[y, z]] != self.meet_tensor[x, y]:
                return False
        return True

    def is_join_semidistributive(self):
        for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            if self.join_tensor[x, y] == self.join_tensor[x, z] and self.join_tensor[x, self.meet_tensor[y, z]] != self.join_tensor[x, y]:
                return False
        return True

    def is_semidistributive(self):
        if self.is_join_semidistributive() and self.is_meet_semidistributive():
            return True
        return False

    def compute_majmin_tensors(self):
        '''
            Compute the majority/minority matrix multipling raws/columns of loe
        '''

        idx_kron = [i*(self.size+1) for i in range(self.size)]

        majority_tensor = torch.kron(self.loe, self.loe)[:, [idx_kron]].reshape(self.size, self.size, self.size)
        minority_tensor = torch.kron(self.loe_transposed, self.loe_transposed)[:, [idx_kron]].reshape(self.size, self.size, self.size)

        return majority_tensor, minority_tensor

    def compute_joinmeet(self):
        '''
            Compute the join/meet matrix multipling loe to majority/minority element rows x columuns and then component-wise
            still to majority/minority element. With these 2 operation a vector which counts the # of majorities grater
            than a majority n is computed and then the join of a pair (a,b) is detached as the majority that as # of majorities
            grater that it = to the corresponding component in the obtained vector.
        '''


        try:
            join_tensor_full = torch.matmul(self.majority_tensor, self.loe)
            join_tensor = (join_tensor_full == 1).nonzero(as_tuple=True)[-1].reshape(self.size, self.size)

            meet_tensor_full = torch.matmul(self.minority_tensor, self.loe_transposed)
            meet_tensor = (meet_tensor_full == 1).nonzero(as_tuple=True)[-1].reshape(self.size, self.size)

            is_a_lattice = True

        except:
            join_tensor = None
            meet_tensor = None
            is_a_lattice = False

        return join_tensor, meet_tensor, is_a_lattice

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

