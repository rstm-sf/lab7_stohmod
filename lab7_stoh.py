import math
import numpy as np
import matplotlib.pyplot as plt

def gen_pos_mol(N, L, n_mol0, dt, A, x0, k1, kp):
	x = [[x0] for i in range(n_mol0)]
	L2 = 2*L
	L_5 = L / 5
	more_r1 = k1 * dt	
	more_r2 = kp * dt * L_5
	tmp = math.sqrt(2*A*dt)

	for k in range(N):
		del_ind = []
		for i in range(len(x)):
			ksi = np.random.uniform(-1,1)
			x_i = x[i]
			x_ij = x_i[len(x_i)-1] + tmp * ksi
			if x_ij < 0.0:
				x_ij = -x_ij
			elif x_ij > L:
				x_ij = L2 - x_ij
			x[i].append(x_ij)
			r1 = np.random.uniform(0,1)
			if r1 < more_r1:
				del_ind.append(i)

		for i in del_ind:
			del x[i]

		r2 = np.random.uniform(0,1)
		if r2 < more_r2:
			r3 = np.random.uniform(0,1)
			x.append([r3*L_5])

	return x

A = 1e-4; L = 1.0; x0 = 0.15; n_mol0 = 100;
dt = 1e-2; N = round(10*60/dt);
k1 = 1e-3; kp = 0.012;

x = gen_pos_mol(N, L, n_mol0, dt, A, x0, k1, kp)

H = 40; h = L / H;
plt.hist([j for i in x for j in i], bins=[i * h for i in range(H+1)])
plt.xlabel('x[mm]')
plt.ylabel('number of molecules')
plt.show()