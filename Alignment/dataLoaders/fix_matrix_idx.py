import h5py
import dill
import numpy as np

h5path = '/home/data_share/FBA/fall19/data/matrix0/middle_2_3_matrix.h5'
h5path_n = '/home/data_share/FBA/fall19/data/matrix0/middle_2_3_matrix_n.h5'

data = h5py.File(h5path, 'r')
data_n = h5py.File(h5path_n, 'w')

sc_dim = 500
pc_dim = 6931

data_n.create_dataset('matrix', (0, pc_dim), maxshape=(None, pc_dim))
data_n.create_dataset('sc_idx', (0, 3), maxshape=(None, 3)) # (i, j, k) -> f['matrix'][i:j, 0:k]

def write_incre_h5(dataset, datapoint):
    #print(datapoint.shape)
    dataset.resize(dataset.shape[0]+datapoint.shape[0],axis=0)
    dataset[-datapoint.shape[0]:,:] = datapoint

idc = np.arange(1410) * 3

write_incre_h5(data_n['sc_idx'], data['sc_idx'][idc, :])

print(data['sc_idx'].shape, data_n['sc_idx'].shape)

m, n = data['matrix'].shape
print(m, n)
print(data['matrix'].shape)

for i in np.arange(int(m/1000)):
    dp = data['matrix'][i*1000:i*1000+1000, :]
    #dp = dp.reshape(1, -1)
    if i % 50 == 0:
        print(i)
        print(data_n['matrix'].shape)
    write_incre_h5(data_n['matrix'], dp)

i = int(m/1000)
dp = data['matrix'][i*1000:, :]
write_incre_h5(data_n['matrix'], dp)

print(data_n['matrix'].shape)

data.close()
data_n.close()
