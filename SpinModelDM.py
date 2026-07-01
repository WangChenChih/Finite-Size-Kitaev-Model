import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from numpy import linalg as LA
from itertools import permutations, combinations
from scipy.special import comb
from scipy.special import factorial
from scipy import integrate
import os

class SpinSystem:

    def __init__(self, 
                 width, height, 
                 Jz, Jx, Jy, 
                 CorrTableCalculate=False, 
                 tablefilename='table'):
        '''
        width and height of the chosen geometric region. The shape of the region is restricted 
        to be rectangular. 
        
        The physical objects of the model are spin-1/2 local spins. The lattice is honeycomb lattice.

        If CorrTableCaculate=True, then the programm will calculate a new table. But if the 
        boolean value is False, then it will directly read and load the table with filename 
        'tablefilename'. The default tablefilename 'table' is just a random name, it will be changed 
        after the table is made or after the code found that the table has already existed. 
        '''
        self.width = int(width)
        self.height = int(height)
        self.N = int(width*height) # total number of sites
        self.dim = int(2**(width*height)) # 2^N, the dimension of the total Hilbert space
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

        self.I = np.eye(2, dtype = np.complex64)
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
        label = self.width*layer + site

        return label
    
    def dictionary_n_to_c(self, label:int)->tuple[int, int, int]:
        '''
        Input the label of the site (an integer) then output the coordinate representation of 
        the site (\vec{x}, \mu). 

        Components of \vec{x} are integers m1 and m2 such that the position of the Bravais 
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
        label = self.dictionary_l_to_n(layer=layer, site=site)
        m1, m2, mu = self.dictionary_n_to_c(label=label)

        return m1, m2, mu
    
    
    ################### operators ###################

    def one_spin(self,length:int, op:np.ndarray, site:int)->np.ndarray:
        """
        Put an operator "op" on "site"
        """
        L = length
        sI = self.sI

        A = None
        for j in range(L):
            Aj = op if j == site else sI
            A = Aj if A is None else np.kron(A, Aj) # A being None means we are at the first site, so we don't have anyone to kron prod yet.
        
        return A

    def two_spin(self, length:int, site1:int, op1:np.ndarray, site2:int, op2:np.ndarray)->np.ndarray:
        """
        Put "op1" and "op2" on "site1" and "site2" respectively
        """
        L = length
        sI = self.sI

        A = None
        for site in range(L):
            if site == site1:
                Aj = op1
            elif site == site2:
                Aj = op2
            else:
                Aj = sI
            
            A = Aj if A is None else np.kron(A, Aj)
        
        return A
    
    def dimer(self, n1: int, n2:int)->np.ndarray:
        """
        Input two nearest neighboring sites n1 and n2, and then return the dimer operator in the Kitaev model \dimer(n1,n2)
        """
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