import numpy as np
import matplotlib.pyplot as plt
import csv

M = 100 # the number of trials 100
K = 3000 # maximum time delay

alpha = 0.90
delta = 0.010

for n in [100,200,400,800,1600]:
    mkav = np.zeros(K)
    mkvar = np.zeros(K)
    mk0av = np.zeros(K)
    mk0var = np.zeros(K)
    for i in range(M):
        f = open("ron_mk_a{:.2f}_d{:.3f}_n{:d}_tr{:d}.txt".format(alpha,delta,n,i),"r")
        rd = csv.reader(f,delimiter=" ")
        
        for j, row in enumerate(rd):
            _, mk, mk0 = row
            x = float(mk)
            y = float(mk0)
            mkav[j] += x
            mk0av[j] += y
            mkvar[j] += x*x
            mk0var[j] += y*y
        f.close()
    mkav /= M
    mk0av /= M
    mkvar = mkvar/M - mkav*mkav
    mk0var = mk0var/M - mk0av*mk0av
    
    f = open("ron_mkav_a{:.2f}_d{:.3f}_n{:d}.txt".format(alpha,delta,n),"w")
    for j in range(K):
        f.write("{:d} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(j,mkav[j],mkvar[j],mk0av[j],mk0var[j]))
    f.close()