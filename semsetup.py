def zwgll(p):
    ''' Generate coordinates and weights for GLL points '''
    import numpy as np
    import scipy as sp
    import scipy.linalg as spla
    
    n = p+1
    z = np.zeros((n,))
    w = np.zeros((n,))
    z[0] = -1.
    z[p] = 1.
    if p < 1:
        print('zeroth order not supported')
        return((z,w))
    elif p == 2:
        z[1] = 0. # fill in one point
    elif p > 2:
        M = np.zeros((p-1,p-1))
        for i in np.arange(p-2):
            M[i,i+1] = (1./2.)*np.sqrt(\
                    float( ((i+1)*(i+1+2))/((i+1+1./2.)*(i+1+3./2.)) ))
            M[i+1,i] = M[i,i+1]
            
        [D,vr] = spla.eigh(M)
        z[1:p] = np.copy(np.sort(D))
    
    w[0] = 2./float(p*n)
    w[p] = w[0]
    for i in np.arange(1,p):
        x = z[i]
        z0 = 1.
        z1 = x
        for j in np.arange(1,p):
            z2 = x*z1*(2.*j + 1.)/(j+1.)-z0*j/(j+1.)
            z0 = z1
            z1 = z2
        w[i] = 2./(p*n*(z2*z2))
    return (z,w)

def fd_weights_full(xx,x,m):
    ''' Appendix of
     A Practical Guide to Pseudospectral Methods, B. Fornberg
     Cambridge Univ. Press, 1996.
    
    xx : point at which derivative to be evaluated
    x  : 1d vector
    m  : highest derivative order
    '''
    import numpy as np
    n1 = x.shape[0]
    n  = n1-1
    m1 = m+1
    c1 = 1.
    c4 = x[0] - xx
    c  = np.zeros((n1,m1))
    c[0,0] = 1.
    for i in np.arange(1,n1): # 1,...,n-1,n
        mn = min(i,m)
        c2 = 1.
        c5 = c4
        c4 = x[i] - xx
        for j in np.arange(0,i): # 0,1,...,i-1
            c3 = x[i]-x[j]
            c2 = c2*c3
            for k in np.arange(mn,0,-1): # mn,mn-1,...,1
                c[i,k]=c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
            c[i,0] = -c1*c5*c[i-1,0]/c2
            for k in np.arange(mn,0,-1):
                c[j,k]=(c4*c[j,k]-k*c[j,k-1])/c3
            c[j,0] = c4*c[j,0]/c3
        c1 = c2
    return c

def dhat(z):
    ''' 1D derivative matrix '''
    n1 = z.shape[0]
    import numpy as np
    dh = np.zeros((n1,n1))
    dw = np.zeros((n1,2))
    for i in np.arange(0,n1):
        dw = fd_weights_full(z[i],z,1)
        dh[:,i] = np.copy(dw[:,1])
    dh = dh.T
    return dh

def semhat(p):
    ''' Compute 1D sem stiffness, mass, adv matrices '''
    [z,w] = zwgll(p)
    import numpy as np
    bh = np.diag(w)
    dh = dhat(z)
    ah = dh.T.dot(bh.dot(dh))
    ch = bh.dot(dh)
    return (ah,bh,ch,dh,z,w)

def interp_mat(xo,xi):
    ''' Compute 1D SEM Interpolation matrix
    xo: output points
    xi: input points (on GLL basis)
    '''
    import numpy as np
    no = xo.ravel().shape[0]
    ni = xi.ravel().shape[0]
    Jh = np.zeros((ni,no))
    w  = np.zeros((ni,2))
    for i in range(no):
        w = fd_weights_full(xo[i],xi,1)
        Jh[:,i] = w[:,0]
    Jh = Jh.T
    return Jh

def semhatf(p):
    ''' Compute 1D SEM Stiffness, Full Mass, Adv matrices; '''
    import numpy as np
    po = p+1
    [z,w]   = zwgll(p)
    [zo,wo] = zwgll(po)
    jh = interp_mat(zo,z)
    bh = jh.T.dot(np.diag(wo)).dot(jh)
    dh = dhat(z)
    ah = dh.T.dot(bh.dot(dh))
    ch = bh.dot(dh)
    return (ah,bh,ch,dh,z,w)

def semqr(E,n,bc):
    ''' Generate SEM assembly and restriction matrices
    E: number of elements, int
    n: polynomial order, int
    bc: boundary conditions, char*2
    '''
    import numpy as np
    import scipy as sp
    n1 = n + 1
    if(bc=='DD' or bc=='dd'):
        Q = sp.zeros((E*n1, E*n+1))
        i = 0
        j = 0
        for ie in range(E):
            Q[i:(i+n+1),j:(j+n+1)] = np.eye(n1)
            i = i + n1
            j = j + n
        R = sp.eye(E*n+1)
        R = R[1:-1,:]
    elif(bc=='nn'):
        Q = sp.zeros((E*n1, E*n+1))
        i = 0
        j = 0
        for ie in range(E):
            Q[i:(i+n+1),j:(j+n+1)] = np.eye(n1)
            i = i + n1
            j = j + n
        R = sp.eye(E*n+1)
    elif(bc=='PP' or bc=='pp'):
        Q = sp.zeros((E*n1, E*n))
        i = 0
        j = 0
        for ie in range(E):
            if(ie==E-1):
                Q[i:(i+n),j:(j+n)] = np.eye(n)
                Q[i+n,0] = 1. # tail to head
            else:
                Q[i:(i+n+1),j:(j+n+1)] = np.eye(n1)
                i = i + n1
                j = j + n
        R = sp.eye(E*n)
    else:
        print('semqr:: Unknow boundary conditions! %s'%bc)

    return Q,R

def semx(xa,xb,E,n,z):
    '''Generate 1D SEM point coordinates without repetition
    xa, xb: min and max, floats
    E: number of elements
    n: polynomial order
    z: reference GLL points, array of (n+1,1)
    '''
    import numpy as np
    import scipy as sp

    le = (xb-xa)/float(E)
    x = np.zeros((E*(n)+1,))
    i = 0
    for ie in range(E):
        x[i:(i+n+1)] =  xa + ie*le + 0.5*le*(z+1.)
        i = i + n

    return x

