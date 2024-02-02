import numpy as np
from scipy.sparse import coo_matrix

class GridState:

    def __init__(self, dim, length, dtype="real"):
        
        self.dim = dim
        self.length = length
        self.n_sites = length**dim
        self.n_neighbours = 2**dim
        self._init_neighbours()
        
        # spin are real valued or int valued
        self.dtype = dtype
        self._init_sites(self.dtype)
        

    def _init_sites(self, dtype):
        # self.sites = np.ones(shape=(self.n_sites), dtype=np.float)
        if dtype == "int":
            # generate sequence of [0,1,0,1,0,0...]
            self.sites = np.random.randint(2, size=self.n_sites)
        if dtype == "real" or dtype== "float":
            self.sites = np.random.randn(self.n_sites)

    
    def _get_flatten_index(self, grid_index):
        dim_array  = np.arange(self.dim)
        # index  = sum(L**d * i[d])
        index = np.sum(np.power(self.length, dim_array) * grid_index)
        return index

    def _get_grid_index(self, flatten_index):
        
        index = np.zeros(shape=(self.dim),dtype=np.int)
        for d in range(self.dim):
            index[d] = flatten_index % self.length
            flatten_index = flatten_index // self.length

        return index


    def _init_neighbours(self):
        self.neighbours = np.zeros(shape=(self.n_sites, self.n_neighbours),dtype=np.int)
        for d in range(self.dim):
            for i in range(self.n_sites):
                grid_index = self._get_grid_index(i)
                neighbour1_index =  grid_index.copy()
                neighbour1_index[d] = (grid_index[d]-1)%self.length
                neighbour2_index =  grid_index.copy()
                neighbour2_index[d] = (grid_index[d]+1)%self.length
                self.neighbours[i,2*d] = self._get_flatten_index(neighbour1_index)
                self.neighbours[i,2*d+1] = self._get_flatten_index(neighbour2_index)


    def get_adjacent_matrix(self):
        row = []
        col = []
        data = []
        for site in range(self.n_sites):
            for i in range(self.n_neighbours):
                row.append(site)
                col.append(self.neighbours[site,i])
                data.append(1)
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        adjacent = coo_matrix((data, (row, col))).toarray()
        return adjacent

    def get_state(self):
        return self.sites

    def set_state(self, new_state):
        self.sites = new_state

