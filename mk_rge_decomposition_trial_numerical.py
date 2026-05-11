import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

M = 20 # the number of trials 100
K = 200 # maximum time delay

T_trans = 1000

alpha = 0.90
delta = 0.010

tmax = int(np.log(1e-12)/np.log(alpha)) + 1 # maximum power of W

for n in [200,400,800]:
    for i in range(M):
        w = np.random.normal(0, 1/np.sqrt(n), (n,n))
        a, b = LA.eig(w)
        a = list(np.abs(a))
        a.sort()
        l1 = a[-1]
        w = w.T*np.sqrt(alpha)/l1
        
        u = ortho_group.rvs(n)
        v = u[:,0]
        
        #numerical
        for T in [5000,10000,20000]:
            s = np.random.normal(0,1,T+T_trans)
            x = np.random.normal(0,1,n)
            for j in range(T_trans):
                z = np.random.normal(0,np.sqrt(delta),n)
                x = np.dot(x,w) + s[j]*v + z
                
            p = [np.zeros(n) for j in range(K)]
            for jj in range(K):
                p[jj] = p[jj] + s[T_trans-1-jj]*x
            xx = x
            for j in range(T-1):
                z = np.random.normal(0,np.sqrt(delta),n)
                x = np.dot(x,w) + s[j+T_trans]*v + z
                
                xx = np.vstack((xx,x))
                for jj in range(K):
                    p[jj] = p[jj] + s[j+T_trans-jj]*x
            
            for j in range(K):
                p[j] /= T
            
            c = np.dot(xx.T,xx)/T
            
            a, b = LA.eigh(c)
            idx = a.argsort()
            a = a[idx]
            b = b[:,idx]
            
            f = open("rge_ev_numerical_a{:.2f}_d{:.3f}_n{:d}_t{:d}_tr{:d}.txt".format(alpha,delta,n,T,i),"w")
            for j in range(n):
                f.write("{:.15f}\n".format(a[n-1-j]))
            f.close()
            
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
            for j in range(K):
                q = np.array(p[j])
                r = q.reshape(1,n)
                mk[j] = np.dot(r,np.dot(d,q))
                mk0[j] = np.dot(r,q)**2/np.dot(r,np.dot(c,q))
                
                if mk[j] < 1e-12:
                    break
            kmax = j
            
            f = open("rge_mk_numerical_a{:.2f}_d{:.3f}_n{:d}_t{:d}_tr{:d}.txt".format(alpha,delta,n,T,i),"w")
            for j in range(kmax):
                f.write("{:d} {:.15f} {:.15f}\n".format(j,mk[j],mk0[j]))
            f.close()
            
            mc = np.sum(mk)
            mclb = np.sum(mk0)
            print(n,T,i,dim,kmax,mc,mclb)
            print((np.sum(a)/np.sum(a*a))*(np.sum(a)/n))
            
        # theoretical
        p = v.reshape(n,1)
        q = p.reshape(1,n)
        
        w = w.T
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
        
        f = open("rge_ev_theoretical_a{:.2f}_d{:.3f}_n{:d}_tr{:d}.txt".format(alpha,delta,n,i),"w")
        for j in range(n):
            f.write("{:.15f}\n".format(a[n-1-j]))
        f.close()
        
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
        
        f = open("rge_mk_theoretical_a{:.2f}_d{:.3f}_n{:d}_tr{:d}.txt".format(alpha,delta,n,i),"w")
        for j in range(kmax):
            f.write("{:d} {:.15f} {:.15f}\n".format(j,mk[j],mk0[j]))
        f.close()
        
        mc = np.sum(mk)
        mclb = np.sum(mk0)
        print(n,i,dim,kmax,mc,mclb)
        print((np.sum(a)/np.sum(a*a))*(np.sum(a)/n))

