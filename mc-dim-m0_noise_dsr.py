import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

M = 1

K = 5000 # maximum time delay

n = 800
alpha = 0.9
ep = 1e-1

tmax = int(np.log(1e-12)/np.log(alpha)) + 1 # maximum power of W

m0ss = []
mcss = []
dimss = []

for i in range(M):
    f = open("mc-dim-m0_noise_dsr_a{:.2f}_e{:.2f}_n{:d}_tr{:d}.txt".format(alpha,ep,n,i),"w")
    
    x = ortho_group.rvs(n)
    
    v = x[:,0]
    w = np.zeros((n,n))
    for k in range(n-1):
        p = x[:,k+1].reshape(n,1)
        q = x[:,k].reshape(1,n)
        w += np.dot(p,q)
    w = np.sqrt(alpha)*w
    
    m0s = []
    mcs = []
    dims = []
    if i == 0:
        deltas = []
    
    delta = 0
    while delta < 2*ep*(1-alpha)+ep*(1-alpha)*1e-2:
        p = v.reshape(n,1)
        q = p.reshape(1,n)
        
        ww = w
        c = np.dot(p,q)
        cn = np.identity(n)
        for j in range(tmax):
            p = np.dot(w,p).reshape(n,1)
            q = p.reshape(1,n)
            
            c += np.dot(p,q)
            cn += np.dot(ww,ww.T)
            
            ww = np.dot(ww,w)
        
        c = c + delta*cn
        
        a, b = LA.eigh(c)
        idx = a.argsort()
        a = a[idx]
        b = b[:,idx]
        
        k = 0 #effective dimension
        for j in range(n):
            if a[j] > ep:
                k += 1
        
        d = np.zeros(n)
        for j in range(n):
            if j < n-k:
                d[j] = 0.0
                for jj in range(n):
                    b[jj,j] = 0.0
            else:
                d[j] = 1.0/a[j]
        d = np.dot(b,np.dot(np.diag(d),b.T))
        
        mk = np.zeros(K)
        mk[0] = np.dot(v,np.dot(d,v))
        mc = mk[0]
        p = v.reshape(n,1)
        q = p.reshape(1,n)
        for j in range(K-1):
            if mk[j] < 1e-12:
                break
            
            p = np.dot(w,p).reshape(n,1)
            q = p.reshape(1,n)
            mk[j+1] = np.dot(q,np.dot(d,p))
            mc += mk[j+1]
        
        m0s.append(mk[0])
        mcs.append(mc)
        dims.append(k)
        
        f.write("{:.15f} {:.15f} {:d} {:.15f}\n".format(delta,mc,k,mk[0]))
        print(i,delta,mc,k,mk[0])
        
        if i == 0:
            deltas.append(delta)
        delta = delta+2*ep*(1-alpha)*1e-2
    m0ss.append(m0s)
    mcss.append(mcs)
    dimss.append(dims)
    f.close()

m0ss = np.array(m0ss)
mcss = np.array(mcss)
dimss = np.array(dimss)
f = open("mc-dim-m0_noise_dsr_a{:.2f}_e{:.2f}_n{:d}.txt".format(alpha,ep,n),"w")
for i,delta in enumerate(deltas):
    f.write("{:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(delta,np.mean(mcss[:,i]),np.var(mcss[:,i]),np.mean(dimss[:,i]),np.var(dimss[:,i]),np.mean(m0ss[:,i]),np.var(m0ss[:,i])))
f.close()
