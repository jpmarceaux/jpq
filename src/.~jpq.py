import numpy as np

def matrize(vec):
    """
    convention: stack rows
    """
    dim = np.sqrt(vec.size)
    assert abs(dim - int(dim)) < 1e-9, "ERROR::matrize(vec); Vector not of square length."
    dim = int(dim)
    
    arry = []
    for d1 in range(dim):
        arry.append([])
        for d2 in range(dim):
            arry[d1].append(vec[d1*dim+d2])
    return np.array(arry)
#xmatrize

def vectorize(mat):
    """
    convention: stack columns 
    """
    dims = mat.shape
    assert len(dims) == 2, "input matrix not a square matrix"
    Dr = dims[0]
    Dc = dims[1]
    
    vecout = np.zeros((Dc*Dr), dtype=complex)
    for dr in range(Dr):
        for dc in range(Dc):
            vecout[dr*Dr + dc] = mat[dr, dc]
        #xfor
    #xfor 
    return vecout
    
def num2base(n, b):
    if n == 0:
        
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]
#xnum2base

def dagger(mat):
    """
    Dagger operation = complex conjugate & transpose
    """
    return np.conj(np.transpose(mat))
#end dagger


def TrX(p, sys, dim):
    """
    Calculate the partial trace over the specified 
    
    credit: http://www.dr-qubit.org/matlab/TrX.m
    """
    
    # check argument errors
    if(any(s > len(dim)-1 for s in sys) or any(s < 0 for s in sys)):
        raise ValueError("Invalid subsystem in SYS")
    if( (len(dim) == 1 and (len(p)/dim)%1 != 0) or
        len(p) != np.prod(dim) ):
        raise ValueError("Size of state PSI inconsistent with DIM")
    
    # remove singleton dimensions
    dim = [d for d in dim if d > 1]
    
    # calculate systems, dimensions, etc...
    n = len(dim)
    rdim = dim[::-1]
    keep = [r for r in range(n)]
    for s in sys:
        keep.remove(s)
    
    dimtrace = 1
    for s in sys:
        dimtrace = dimtrace*dim[s]
    dimkeep = len(p)/dimtrace
    
    # TODO: add support for state vectors
    
    # density matrix trace
    
    # reshape density matrix into tensor with one row and one column index
    # for each subsystem, permute traced subssytem indices to the end, 
    # reshape again so that first two indices are row and column
    # multi-indices for kept sybsystem and third index is a flattened index
    # for traced subsystems, then sum third indeix over "diagonal" entries 
    
    preperm = keep[::-1]+[k-n for k in keep[::-1]]+sys+[s-n for s in sys]
    perm = [n-pp-1 for pp in preperm]
    x = np.reshape((np.reshape(p, rdim+rdim)).transpose(perm), [int(dimkeep), int(dimkeep), int(dimtrace**2)])
    diag = np.arange(0, dimtrace**2, dimtrace)
    x = np.sum(x[:, :, diag], 2)
    return 
#!TrX


def UMEvec(dim):
    """unormalized, maximally entanged ket
    
    The form of UMEvec is ones at |i>|i> and zero elsewhere
    """
    psi = np.zeros((dim**2), complex)
    for n in range(0, dim):
        psi[n*(dim+1)] = 1
    #xfor
    return psi
#@UMEvec

def UMEmat(dim):
    """unormalized, maximally entanged density matrix"""
    psi = UMEvec(dim)
    return np.outer(psi, psi)
#@UMEmat


def compose(chnl1, chnl2):
    return Channel(
        chnl1.rep("lop")@chnl2.rep("lop"), 
        "lop"
    )
#!compose

import numpy as np
from numpy import sqrt

def matrize(vec):
    """
    convention: stack rows
    """
    dim = np.sqrt(vec.size)
    assert abs(dim - int(dim)) < 1e-9, "ERROR::matrize(vec); Vector not of square length."
    dim = int(dim)
    
    arry = []
    for d1 in range(dim):
        arry.append([])
        for d2 in range(dim):
            arry[d1].append(vec[d1*dim+d2])
    return np.array(arry)
#xmatrize

def vectorize(mat):
    """
    convention: stack columns 
    """
    dims = mat.shape
    assert len(dims) == 2, "input matrix not a square matrix"
    Dr = dims[0]
    Dc = dims[1]
    
    vecout = np.zeros((Dc*Dr), dtype=complex)
    for dr in range(Dr):
        for dc in range(Dc):
            vecout[dr*Dr + dc] = mat[dr, dc]
        #xfor
    #xfor 
    return vecout
    
def num2base(n, b):
    if n == 0:
        
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]
#xnum2base

def dagger(mat):
    """
    Dagger operation = complex conjugate & transpose
    """
    return np.conj(np.transpose(mat))
#end dagger


def TrX(p, sys, dim):
    """
    Calculate the partial trace over the specified 
    
    credit: http://www.dr-qubit.org/matlab/TrX.m
    """
    
    # check argument errors
    if(any(s > len(dim)-1 for s in sys) or any(s < 0 for s in sys)):
        raise ValueError("Invalid subsystem in SYS")
    if( (len(dim) == 1 and (len(p)/dim)%1 != 0) or
        len(p) != np.prod(dim) ):
        raise ValueError("Size of state PSI inconsistent with DIM")
    
    # remove singleton dimensions
    dim = [d for d in dim if d > 1]
    
    # calculate systems, dimensions, etc...
    n = len(dim)
    rdim = dim[::-1]
    keep = [r for r in range(n)]
    for s in sys:
        keep.remove(s)
    
    dimtrace = 1
    for s in sys:
        dimtrace = dimtrace*dim[s]
    dimkeep = len(p)/dimtrace
    
    # TODO: add support for state vectors
    
    # density matrix trace
    
    # reshape density matrix into tensor with one row and one column index
    # for each subsystem, permute traced subssytem indices to the end, 
    # reshape again so that first two indices are row and column
    # multi-indices for kept sybsystem and third index is a flattened index
    # for traced subsystems, then sum third indeix over "diagonal" entries 
    
    preperm = keep[::-1]+[k-n for k in keep[::-1]]+sys+[s-n for s in sys]
    perm = [n-pp-1 for pp in preperm]
    x = np.reshape((np.reshape(p, rdim+rdim)).transpose(perm), [int(dimkeep), int(dimkeep), int(dimtrace**2)])
    diag = np.arange(0, dimtrace**2, dimtrace)
    x = np.sum(x[:, :, diag], 2)
    return 
#!TrX


def UMEvec(dim):
    """unormalized, maximally entanged ket
    
    The form of UMEvec is ones at |i>|i> and zero elsewhere
    """
    psi = np.zeros((dim**2), complex)
    for n in range(0, dim):
        psi[n*(dim+1)] = 1
    #xfor
    return psi
#@UMEvec

def UMEmat(dim):
    """unormalized, maximally entanged density matrix"""
    psi = UMEvec(dim)
    return np.outer(psi, psi)
#@UMEmat


def compose(chnl1, chnl2):
    return Channel(
        chnl1.rep("lop")@chnl2.rep("lop"), 
        "lop"
    )
#!compose


class State:
    """
    Type to store quantum states
    
    State is stored as a density matrix always 
    
    if we want the associated ket, we just project the density matrix onto the standard basis
        if state is pure, then the ket is properly normalized
    """
    def __init__(self, arrin):
        # check type of input array and then make density matrix
        assert type(arrin) is type(np.array([])), "input not a numpy array"
        assert len(arrin.shape) <= 2, "input array of dim > 2"
        if len(arrin.shape) == 1:
            self.dmat = np.outer(arrin, np.conj(arrin))
        else:
            self.dmat = arrin
            
    def purity(self):
        return np.trace(self.dmat@self.dmat)
    
    def getKet(self):
        # TODO
        return
#xState

class Operator:
    """
    Type to represent quantum operator 
    """
    def __init__(self, matin):
        # assure the input is either a matrix or 2d numpy array
        assert (type(matin) is type(np.array([]))), "input not a numpy array"
        assert len(matin.shape) == 2, "input not a 2d array"
        assert matin.shape[0] == matin.shape[1], "input array not square"
        self.mat = matin
#xOperator

def cnum2str(cnum, prec=3):
    """Convert cnum to string with given precision"""
    t1 = f'{cnum.real:.{prec}f}'
    t2 = f'{cnum.imag:.{prec}f}'
    return '(' + t1 + ', ' + t2 + ')'
#xcnum2str

def cmat2str(cmat, prec=3):
    strout = ""
    for row in cmat:
        strout += '['
        for c in row:
            strout += cnum2str(c, prec)
            strout += ' '
        #xfor 
        strout += ']\n'
    #xfor
    return strout
#xcmat2str

def mat2html(cmat, prec=3):
    strout = ""
    for row in cmat:
        strout += '['
        for c in row:
            strout += cnum2str(c, prec)
            strout += ' '
        #xfor 
        strout += ']<br>'
    #xfor
    return strout
#xcmat2str

def cmat2html(cmat, prec=3):
    strout = ""
    dims = cmat.shape
    dr = int(sqrt(dims[0]))
    dc = int(sqrt(dims[1]))
    for rdx, row in enumerate(cmat):
        cnt = 0
        strout += '|'
        for idx, c in enumerate(row): 
            strout += cnum2str(c, prec)
            cnt += 1
            if cnt%dc == 0 and idx != len(row) -1:
                strout += '| |'
        #endfor
        strout += "|</br>"
        if rdx%dr == 1:
            strout += "</br>"
    #endfor
    return strout
#!cmat2htmal
    

def cten2html(cten, prec=3):
    return cmat2html(cten2cmat(cten), prec) 
#xcten2html

def cmat2cten(cmat):
    """
    convert a dim**2 x dim**2 matrix into 4d numpy tensor
    """
    dsq = cmat.shape[0]
    assert dsq == cmat.shape[1]     # square matrix
    dim = np.sqrt(dsq)
    assert (dim - int(dim)) < 1e-9   # square number
    dim = int(dim)
    
    cten = np.zeros((dim, dim, dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            cten[i, j, :, :] = (
                cmat[(i*dim):((i+1)*dim), (j*dim):(j+1)*dim]
            )
        #x
    #x
    return cten

def cten2cmat(cten):
    """
    convert choi tensor into 2D matrix
    """
    assert len(cten.shape) == 4
    dim = int(cten.shape[0])
    cmat = np.zeros((dim**2, dim**2), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            cmat[(i*dim):((i+1)*dim), (j*dim):(j+1)*dim] = (
                cten[i, j, :, :]
            )
        #x
    #x
    return cmat

def cmat2kraus(cmat):
    Lambda, Smat = np.linalg.eig(ChoiMat)
    # take transpose of Smat so we are getting eigenvectors; 
    # (transpose of unitary matrix is still unitary)
    # see https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    Smat = Smat.transpose()
    
    KrausSet = []
    for idx, l in enumerate(Lambda):
        svec = Smat[idx]
        lam = np.sqrt(l)
        kvec = lam*svec
        KrausSet.append(matrize(kvec))
    return KrausSet

def kraus2cmat(kset):
    """
    convert kset into choi mat
    """
    dims = kset[0].shape
    assert dims[0] == dims[1], "Kraus operators not square"
    dim = dims[0]
    
    cmout = np.zeros((dim**2, dim**2), dtype=complex)
    Eye = np.eye(dim, dtype=complex)
    Phi = UMEmat(dim)
    
    for K in kset:
        cmout += np.kron(Eye, K)@Phi@np.kron(Eye, dagger(K))
    return cmout

def cten2kraus(cten):
    return cmat2kraus(cten2cmat(cten))

def kraus2cten(kset):
    return cmat2cten(kraus2cmat(kset))

def cmat2lop_kraus(cmat):
    """
    convert choi matrix to linear operator 
    
    calculate by finding kraus set and then using that 
    """
    
    kset = cmat2kraus(cmat)
    dims = (
        int(np.sqrt(cmat.shape[0])), 
        int(np.sqrt(cmat.shape[1]))
    )
    lop = np.zeros(dims, dytpe=complex)
    for K in kset:
        lop += np.kron(K, dagger(K))
    #xfor 
    return lop
#xcmat2lop_krays

def cten2lop(cten):
    """
    convert choi tensor to linear operator 
    
    calculate by vectorizing blocks 
        using row-stacking convention
    """
    
    dims = cten.shape
    
    dx = dims[0]*dims[2]
    dy = dims[1]*dims[3]
    
    assert dx == dy, "choi matrix not square"
    L = sqrt(dx)
    assert abs(L-int(L)) < 1e-9, "cmat not of sqaure dimensions"
    L = int(L)
    
    lop = np.zeros((dx, dy), dtype=complex)
    for i in range(L):
        for j in range(L):
            lop[i*L+j, :] = vectorize(cten[i,j,:,:])
        #xfor
    #xfor    
    return lop
#xcmat2lop

def lop2cten(lop):
    """
    convert linear operator to choi matrix
    
    calculate by calculating K(Phi)|i>|j> and matrizing, i.e., 
        matrizing the row i*D+j
    """
    dims = lop.shape
    assert dims[0] == dims[1], "Lop not square"
    L = sqrt(dims[0])
    assert abs(L-int(L)) < 1e-9, "Lop not of square dimension"
    L = int(L)
    
    cten = np.zeros((L,L,L,L), dtype=complex)
    for i in range(L):
        for j in range(L):
            cij = matrize(lop[i*L+j, :])
            cten[i,j,:,:] = cij
        #xfor
    #xfor    
    return cten
#xlop2cten

def lop2cmat(lop):
    return cten2cmat(lop2cten(lop))
#xlop2cmat

def cmat2lop(cmat):
    return cten2cmat(cmat2lop(cmat))
#xcmat2lop

class Channel:
    """
    Class object to represent a quantum channel 
    
    Stores only the Choi tensor of the channel in the standard basis 
    
    Internal members:
    _cten, 
    """
    def __init__(self, data, rep):
        if rep == "cten":
            self._cten = data
        elif rep == "cmat":
            self._cten = cmat2cten(data)
        elif rep == "lop":
            self._cten = lop2cten(data)
        elif rep == "kraus":
            self._cten = kraus2cten(data)
    #x__init__
    
    def __str__(self):
        return "Channel with choi dimensions (%i, %i, %i, %i)" % self._cten.shape
    
    def apply(self, X):
        """
        use ChoiTensor and sum based on amplitudes of input X in standard basis 
        """
        assert(
            (type(X) is State) or 
            (type(X) is Operator) or
            (type(X) is np.array) or
            (type(X) is np.ndarray) 
        ), "X not state, operator, or numpy array"
        assert X.shape == self._cten[0,0,:,:].shape, "dimensional mismatch"
        dims = X.shape
        xout = np.zeros(dims, dtype=complex)
        for i in range(dims[0]):
            for j in range(dims[1]):
                xout += X[i,j]*self._cten[i,j,:,:]
            #xfor
        #xfor 
        return xout
    #xChannel.apply
        
    def rep(self, crep):
        if crep == "cten":
            return self._cten
        elif crep == "cmat":
            return cten2cmat(self._cten)
        elif crep == "lop":
            return cten2lop(self._cten)
        elif crep == "kraus":
            return cten2kraus(self._cten)
    #xChannel.getRep
    
    def html(self, filename, prec=3):
        """make html representation of channel with given precision"""
        file = ""
        file += "<h1> Quantum channel with dimensions(%i, %i, %i, %i) </h1>" % self._cten.shape
        # print transfer matrix
        file += "<h2> Transfer Matrix: </h2>"
        file += mat2html(self.rep("lop"), prec)
        file += "<h2> Choi Tensor: </h2>"
        file += cten2html(self._cten, prec)
        
        with open(filename, "w") as fileout:
            fileout.write(file)
    #xChannel.html
#@Channel

