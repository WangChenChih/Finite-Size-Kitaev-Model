import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from numpy import linalg as LA
from itertools import permutations, combinations
import scipy.sparse as sp
from scipy.special import comb, factorial
from scipy import integrate
import os

class SpinSystem:

    def __init__(self, 
                 width:int=4, height:int=4, 
                 Jz:float=1, Jx:float=1, Jy:float=1, 
                 CorrTableCalculate=False, 
                 tablefilename='table'):
        '''
        width and height of the chosen geometric region. The shape of the region is restricted 
        to be rectangular. 
        
        The phyIcal objects of the model are spin-1/2 local spins. The lattice is honeycomb lattice.

        If CorrTableCaculate=True, then the programm will calculate a new table. But if the 
        boolean value is False, then it will directly read and load the table with filename 
        'tablefilename'. The default tablefilename 'table' is just a random name, it will be changed 
        after the table is made or after the code found that the table has already existed. 
        '''
        self.width = int(width)
        self.height = int(height)
        self.N = int(width*height) # total number of sites
        self.dim = int(2**(width*height)) # 2^N, the dimenIon of the total Hilbert space
        self.Np = 0 # which will be calculated by the function PlaquetteConstruction

        self._Jx = Jx
        self._Jy = Jy
        self._Jz = Jz

        self.CorrTableExist = not CorrTableCalculate # it will be changed to True after calculation is done
        self.TableName = tablefilename
        if self.CorrTableExist:
            self.corrtable = np.load(tablefilename)
        else:
            self.corrtable = np.array([])   # will be updated if the table exists.
        
        self.corr_dict = {} # will be updated 
        self.corr_dict_exist = False

        self.CanonicalFormTable_dic = {}
        self.CanonicalFormTalbe__dic_exist = False

        self.dimerconfig_exist = False
        self.dimer_configuration_number = 0

        self.I = sp.identity(2, format="csr")
        self.X = np.array([[0,1],[1,0]], dtype = np.complex64)
        self.Y = np.array([[0,-1j],[1j,0]], dtype = np.complex64)
        self.Z = np.array([[1,0],[0,-1]], dtype = np.complex64)

        self.rho = np.array([])
        self.rho_F = np.array([])

        return
    
    @property
    def Jx(self):
        return self._Jx
    @Jx.setter
    def Jx(self, Jx: float):
        self._Jx = Jx
        return
    
    @property
    def Jy(self):
        return self._Jy
    @Jy.setter
    def Jy(self, Jy: float):
        self._Jy = Jy
        return
    
    @property
    def Jz(self):
        return self._Jz
    @Jz.setter
    def Jz(self, Jz: float):
        self._Jz = Jz
        return
    
    ################### correlation functions ###################

    def sk(self, k1,k2):
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        return 2*np.sqrt(Jx**2 + Jy**2 + Jz**2 + 
                    2*Jx*Jy*np.cos(k1) + 
                    2*Jx*Jz*np.cos(k2) + 
                    2*Jy*Jz*np.cos(k1 - k2))
    
    def Rk(self, k1,k2):
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        return 2*(Jx*np.cos(k2) + Jy*np.cos(k1 - k2) + Jz)
    
    def Ik(self, k1,k2):
        Jx = self.Jx
        Jy = self.Jy
        return 2*(Jx*np.sin(k2) - Jy*np.sin(k1 - k2))
    
    ################### dictionary ###################

    def dictionary_n_to_l(self,label:int)->tuple[int, int]:
        '''
        Input the label of the site (an integer) and then ouput the layer-site representation of 
        the site (layer, site).
        '''
        label = int(label)
        site = label%self.width
        layer = int(label/self.width)

        return layer, site
    
    def dictionary_l_to_n(self, layer:int, site:int)->int:
        '''
        Inverse function of dictionary_n_to_l
        '''
        # periodic boundary condition
        layer, site = layer % self.height, site % self.width

        label = self.width*layer + site

        return label
    
    def dictionary_n_to_c(self, label:int)->tuple[int, int, int]:
        '''
        Input the label of the site (an integer) then output the coordinate representation of 
        the site (\vec{x}, \mu). 

        Components of \vec{x} are integers m1 and m2 such that the poItion of the Bravais 
        lattice site is 
        m1\vec{a1} + m2\vec{a2}
        , where 
        \vec{a1} = (1,0)
        \vec{a2} = (1/2, \sqrt{3}/2) 

        \mu = 0 stands for B-sublattice 
            = 1 stands for A-sublattice
        '''
        mu = label % 2

        layer, site = self.dictionary_n_to_l(label=label)
        
        # site = 2 * sq + sr
        sr = site % 2
        sq = int(site/2)

        m1 = sq + sr    # x = sq + sr
        m2 = layer - sr # y = l - sr

        return m1, m2, mu
    
    def dictionary_l_to_c(self, layer:int, site:int)->tuple[int, int, int]:
        # periodic boundary condition
        layer, site = layer % self.height, site % self.width

        label = self.dictionary_l_to_n(layer=layer, site=site)
        m1, m2, mu = self.dictionary_n_to_c(label=label)

        return m1, m2, mu
    
    def SubLatID_l(self, layer:int, site:int)->int:
        """
        identify the sublattice
        """
        n = self.dictionary_l_to_n(layer=layer, site=site)
        return n%2
    
    ################### foundamental operators ###################
    
    def one_spin(self,length:int, op:np.ndarray, site:int)->np.ndarray:
        """
        Put an operator "op" on "site"
        """
        L = length
        I = self.I

        A = None
        for j in range(L):
            Aj = op if j == site else I
            A = Aj if A is None else sp.kron(A, Aj, format="csr") # A being None means we are at the first site, so we don't have anyone to kron prod yet.
        
        return A

    def two_spin(self, length:int, site1:int, op1:np.ndarray, site2:int, op2:np.ndarray)->np.ndarray:
        """
        Put "op1" and "op2" on "site1" and "site2" respectively
        """
        L = length
        I = self.I

        A = None
        for site in range(L):
            if site == site1:
                Aj = op1
            elif site == site2:
                Aj = op2
            else:
                Aj = I
            
            A = Aj if A is None else sp.kron(A, Aj, format="csr")
        
        return A
       
    ################### string operator ###################

    def dimer(self, n1: int, n2:int)->np.ndarray:
        """
        Input two nearest neighboring sites n1 and n2, and then return the dimer operator in the Kitaev model \dimer(n1,n2)
        """
        # n1 is type-A by default. Thus, if the situation is opposite, then we have to switch it to our setting
        mu1, mu2 = n1%2, n2%2
        if mu1 == 1 and mu2 == 0:
            pass
        elif mu1 == 0 and mu2 == 1:
            n1_temp = n1
            n1 = n2
            n2 = n1_temp
        else:
            raise ValueError('two endpoints of a valid string must be of different types')
        
        l1, s1 = self.dictionary_n_to_l(label=n1)
        l2, s2 = self.dictionary_n_to_l(label=n2)

        # determine the direction of the dimer
        if l2 == l1-1 and s1 == s2:
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.Z, op2=self.Z)
        elif s2 == s1 + 1 and l1 == l2:
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.X, op2=self.X)
        elif s2 == s1 - 1 and l1 == l2:
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.Y, op2=self.Y)
        else:
            raise ValueError("Invalid points for forming a dimer")
        
    def string_op(self, na:int, nb:int)->np.ndarray:
        """
        Generate the string operator of a single string, whose endpoints are na (in sublattice A) and nb (in sublattice B)
        """
        la, sa = self.dictionary_n_to_l(label=na)
        lb, sb = self.dictionary_n_to_l(label=nb)
        
        StringOp = sp.identity(self.dim, format="csr")  # initialize the string operator
        l, s = la, sa   # initialize the moving point

        while l != lb or s != sb:
            if lb == la:
                if sb > sa:
                    l_next, s_next = l, s-1
                else:   #sb < sa
                    l_next, s_next = l, s-1
            elif lb > la:
                if self.SubLatID_l(layer=l, site=s) == 0:   # sublattice B
                    l_next, s_next = l+1, s-1
                elif self.SubLatID_l(layer=l, site=s) == 1:   # sublattice A
                    l_next, s_next = l, s+1
            elif lb < la:
                if self.SubLatID_l(layer=l, site=s) == 0:   # sublattice B
                    l_next, s_next = l-1, s+1
                elif self.SubLatID_l(layer=l, site=s) == 1:   # sublattice A
                    l_next, s_next = l, s+1
            else:
                raise ValueError('invalid algorithm')
            
            n, n_next = self.dictionary_l_to_n(layer=l, site=s), self.dictionary_l_to_n(layer=l_next, site=s_next)
            StringOp = StringOp @ self.dimer(n1=n, n2=n_next)

        return StringOp
    
    def SigmaOp(self, setA:list, setB:list)->np.ndarray:
        """
        Producing the string operator of a general edge configuration, which may contain more than one string. setA consists of the type-A endpoints of the edge configuration, while setB contains the type-B ones. 
        """
        # A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints
        if len(setA) != len(setB):
            raise ValueError('A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints')
        
        # sort the sets such that the elements are arranged in an increasing order
        setA.sort()
        setB.sort()

        Sigma = sp.identity(self.dim, format="csr")   # initialize the string operator

        counting = 0
        for na in setA:
            nb = setB[counting]

            Sigma = Sigma @ self.string_op(na=na, nb=nb)

            counting += 1

        return Sigma
    
    def EndpointSets(self)->list[list[list,list]]:
        """
        output

        pairing = 
        [   [setA1, setB1],
            [setA2, setB2],
        ...
        ]
        """
        pairing = []    # initialize the pairing data

        ### sort lattice sites into A and B sublattices
        A_all = []
        B_all = []
        
        nA = 0
        nB = 0
        for i in np.arange(self.N, dtype=int):
            if i%2 == 0:
                B_all += [i,]
            else:
                A_all += [i,]
        
        for n in range(np.size(A_all)+1):   # loop over number of pairings. n=0 means no pairing, i.e. vacuum: identity matrix. n=2 means 2 pairings, i.e. 4 end points.
            comb_A = np.array(list(combinations(A_all, n)))
            comb_B = np.array(list(combinations(B_all, n)))
            for i in range(comb(np.size(A_all), n, exact=True)):    # loop over all possible choices of n type-A sites
                for j in range(comb(np.size(B_all), n, exact=True)):    # loop over all possible choices of n type-B sites
                    pairing += [[comb_A[i,:], comb_B[j,:]],]
        return pairing
    
