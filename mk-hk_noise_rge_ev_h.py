import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

M = 1#100

K = 5000 # maximum time delay

n = 800
alpha = 0.9
ep = 1e-1

tmax = int(np.log(1e-12)/np.log(alpha)) + 1 # maximum power of W

m0ss = []
mcss = []
dimss = []

h0ss = []
hcss = []
hdimss = []

for i in range(M):
    f = open("mc-dim-m0_noise_rge_a{:.2f}_e{:.2f}_n{:d}_tr{:d}.txt".format(alpha,ep,n,i),"w")
    g = open("ev_noise_rge_a{:.2f}_e{:.2f}_n{:d}_tr{:d}.txt".format(alpha,ep,n,i),"w")
    h = open("hvar_noise_rge_a{:.2f}_e{:.2f}_n{:d}_tr{:d}.txt".format(alpha,ep,n,i),"w")
    
    w = np.random.normal(0, 1/np.sqrt(n), (n,n))
    a, b = LA.eig(w)
    a = list(np.abs(a))
    a.sort()
    l1 = a[-1]
    w = w*np.sqrt(alpha)/l1
    
    #v = np.random.normal(0,1/np.sqrt(n),n)
    x = ortho_group.rvs(n)
    v = x[:,0]
    
    m0s = []
    mcs = []
    dims = []
    
    h0s = []
    hcs = []
    hdims = []
    
    if i == 0:
        deltas = []
    
    delta = 0
    while delta < 100*ep*(1-alpha)*1e-2:#2*ep*(1-alpha)+ep*(1-alpha)*1e-2:
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
        
        g.write("{:d} {:.15f} ".format(i,delta))
        for j in range(n):
            g.write("{:.15f} ".format(a[n-1-j]))
        g.write("\n")
        
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
        
        hk = np.zeros(K)
        hvar = np.zeros(n)
        hc = 0
        
        p = v.reshape(n,1)
        q = p.reshape(1,n)
        kk = 0
        for j in range(K):
            var = np.dot(q,np.dot(c,p))/np.dot(q,p)
            if j < n:
                hvar[j] = var
            
            if var > ep:
                hk[j] = np.dot(q,p)**2/np.dot(q,np.dot(c,p))
                if hk[j] >= 1e-12:
                    hc += hk[j]
                    kk += 1
            
            if j >= n-1 and var <= ep and hk[j] < 1e-12:
                break
            
            p = np.dot(w,p).reshape(n,1)
            q = p.reshape(1,n)
        
        h.write("{:d} {:.15f} ".format(i,delta))
        for j in range(n):
            h.write("{:.15f} ".format(hvar[j]))
        h.write("\n")
        
        m0s.append(mk[0])
        mcs.append(mc)
        dims.append(k)
        
        h0s.append(hk[0])
        hcs.append(hc)
        hdims.append(kk)
        
        f.write("{:.15f} {:.15f} {:.15f} {:d} {:d} {:.15f} {:.15f}\n".format(delta,mc,hc,k,kk,mk[0],hk[0]))
        print(i,delta,mc,hc,k,kk,mk[0],hk[0])
        
        ff = open("mk-hk_noise_rge_a{:.2f}_e{:.2f}_d{:.4f}_n{:d}_tr{:d}.txt".format(alpha,ep,delta,n,i),"w")
        for j in range(K):
            ff.write("{:d} {:.15f} {:.15f}\n".format(j,mk[j],hk[j]))
        ff.close()
        
        if i == 0:
            deltas.append(delta)
        #delta = delta+ep*(1-alpha)*1e-2
        delta = delta+5*ep*(1-alpha)*1e-2
    m0ss.append(m0s)
    mcss.append(mcs)
    dimss.append(dims)
    h0ss.append(h0s)
    hcss.append(hcs)
    hdimss.append(hdims)
    f.close()
    g.close()
    h.close()

m0ss = np.array(m0ss)
mcss = np.array(mcss)
dimss = np.array(dimss)
h0ss = np.array(h0ss)
hcss = np.array(hcss)
hdimss = np.array(hdimss)
f = open("mc-dim-m0_noise_rge_a{:.2f}_e{:.2f}_n{:d}.txt".format(alpha,ep,n),"w")
for i,delta in enumerate(deltas):
    f.write("{:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(delta,np.mean(mcss[:,i]),np.var(mcss[:,i]),np.mean(dimss[:,i]),np.var(dimss[:,i]),np.mean(m0ss[:,i]),np.var(m0ss[:,i])))
f.close()

f = open("hc-hdim-h0_noise_rge_a{:.2f}_e{:.2f}_n{:d}.txt".format(alpha,ep,n),"w")
for i,delta in enumerate(deltas):
    f.write("{:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(delta,np.mean(hcss[:,i]),np.var(hcss[:,i]),np.mean(hdimss[:,i]),np.var(hdimss[:,i]),np.mean(h0ss[:,i]),np.var(h0ss[:,i])))
f.close()