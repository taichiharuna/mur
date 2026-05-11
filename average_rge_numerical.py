import numpy as np
import matplotlib.pyplot as plt
import csv

M = 20 # the number of trials 100
K = 101 # maximum time delay

alpha = 0.90
delta = 0.010

g = open("rge_lbcsav_a{:.2f}_d{:.3f}.txt".format(alpha,delta),"w")

for n in [200,400,800]:
    for T in [5000,10000,20000]:
        mkav = np.zeros(K)
        mkvar = np.zeros(K)
        mk0av = np.zeros(K)
        mk0var = np.zeros(K)
        for i in range(M):
            f = open("rge_mk_numerical_a{:.2f}_d{:.3f}_n{:d}_t{:d}_tr{:d}.txt".format(alpha,delta,n,T,i),"r")
            rd = csv.reader(f,delimiter=" ")
            
            for j, row in enumerate(rd):
                if j == K:
                    break
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
        
        f = open("rge_mkav_numerical_a{:.2f}_d{:.3f}_n{:d}_t{:d}.txt".format(alpha,delta,n,T),"w")
        for j in range(K):
            f.write("{:d} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(j,mkav[j],mkvar[j],mk0av[j],mk0var[j]))
        f.close()
    
    mkav = np.zeros(K)
    mkvar = np.zeros(K)
    mk0av = np.zeros(K)
    mk0var = np.zeros(K)
    for i in range(M):
        f = open("rge_mk_theoretical_a{:.2f}_d{:.3f}_n{:d}_tr{:d}.txt".format(alpha,delta,n,i),"r")
        rd = csv.reader(f,delimiter=" ")
        
        for j, row in enumerate(rd):
            if j == K:
                break
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
    
    f = open("rge_mkav_theoretical_a{:.2f}_d{:.3f}_n{:d}.txt".format(alpha,delta,n),"w")
    for j in range(K):
        f.write("{:d} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(j,mkav[j],mkvar[j],mk0av[j],mk0var[j]))
    f.close()
    
    lbcs = np.zeros(M)
    for i in range(M):
        f = open("rge_ev_theoretical_a{:.2f}_d{:.3f}_n{:d}_tr{:d}.txt".format(alpha,delta,n,i),"r")
        rd = csv.reader(f,delimiter=" ")
        evsum = 0.0
        ev2sum = 0.0
        for row in rd:
            x = float(row[0])
            evsum += x
            ev2sum += x*x
        lbcs[i] = (evsum/ev2sum)*(evsum/n)
        f.close()
    g.write("{:d} {:.15f} {:.15f}\n".format(n,np.mean(lbcs),np.var(lbcs)))
g.close()
    