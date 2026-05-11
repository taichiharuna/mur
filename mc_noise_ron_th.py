import numpy as np

n = 800
alpha = 0.9
ep = 1e-1

mcs = []
deltas = []
delta = 0.0
while delta < 2*ep*(1-alpha):
    
    mc = 0.0
    for k in range(n):
        mu = alpha**k + delta/(1 - alpha)
        if mu > ep:
            mc += alpha**k/mu
    mcs.append(mc)
    deltas.append(delta)
    delta = delta + ep*(1-alpha)*1e-3

f = open("mc_noise_ron_th_a{:.2f}_e{:.2f}_n{:d}.txt".format(alpha,ep,n),"w")
for i,delta in enumerate(deltas):
    f.write("{:.15f} {:.6f}\n".format(delta,mcs[i]))
f.close()