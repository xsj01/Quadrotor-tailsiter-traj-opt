import numpy as np


filename = "traj/rand.npz"
infile = open(filename, 'r')
npzfile = np.load(infile)

traj = npzfile["arr_0"]
u_traj = npzfile["arr_1"]
time_array = npzfile["arr_2"]

print(traj[:10, :6])
print(u_traj.shape)
print(time_array.shape)


a = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
indices = np.where(a > 3)
print(indices[0][0])
