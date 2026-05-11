import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

M = 100 # the number of trials 100

K = 5000 # maximum time delay

alpha = 0.90
delta = 0.010

tmax = int(np.log(1e-12)/np.log(alpha)) + 1 # maximum power of W

for n in [100,200,400,800,1600]:
    g = open("rge_mc_a{:.2f}_d{:.3f}_n{:d}.txt".format(alpha,delta,n),"w")
    for i in range(M):
        w = np.random.normal(0, 1/np.sqrt(n), (n,n))
        a, b = LA.eig(w)
        a = list(np.abs(a))
        a.sort()
        l1 = a[-1]
        w = w*np.sqrt(alpha)/l1
        
        x = ortho_group.rvs(n)
        v = x[:,0]
        
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
        
        dim = 0 #check dimension
        for j in range(n):
            if a[j] > 0:
                dim += 1
        
        d = np.zeros(n)
        for j in range(n):
            if j < n-dim:
                d[j] = 0.0
                for jj in range(n):
                    b[jj,j] = 0.0
            else:
                d[j] = 1.0/a[j]
        d = np.dot(b,np.dot(np.diag(d),b.T))
        
        mk = np.zeros(K)
        mk0 = np.zeros(K)
        
        y = np.dot(v,np.dot(d,v))
        z = np.dot(v,v)**2/np.dot(v,np.dot(c,v))
        mk[0] = y
        mk0[0] = z
        p = v.reshape(n,1)
        q = p.reshape(1,n)
        tmp = mk[0]
        for j in range(K-1):
            if mk[j] < 1e-12:
                break
                
            p = np.dot(w,p).reshape(n,1)
            q = p.reshape(1,n)
            mk[j+1] = np.dot(q,np.dot(d,p))
            mk0[j+1] = np.dot(q,p)**2/np.dot(q,np.dot(c,p))
        kmax = j
        
        f = open("rge_mk_a{:.2f}_d{:.3f}_n{:d}_tr{:d}.txt".format(alpha,delta,n,i),"w")
        for j in range(kmax):
            f.write("{:d} {:.15f} {:.15f}\n".format(j,mk[j],mk0[j]))
        f.close()
        
        mc = np.sum(mk)
        mclb = np.sum(mk0)
        g.write("{:d} {:d} {:d} {:d} {:.15f} {:.15f}\n".format(i,dim,tmax,kmax,mc,mclb))
        print(n,i,dim,tmax,kmax,mc,mclb)
    g.close()

