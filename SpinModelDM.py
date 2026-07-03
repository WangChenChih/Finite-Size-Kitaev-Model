import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
from numpy import linalg as LA
from itertools import permutations, combinations
import scipy.sparse as sp
from scipy.special import comb, factorial
from scipy import integrate
import os
import json

class SpinSystem:

    def __init__(self, 
                 width:int=4, height:int=4, 
                 Jz:float=1, Jx:float=1, Jy:float=1):
        '''
        width and height of the chosen geometric region. The shape of the region is restricted 
        to be rectangular. 
        
        The phyical constituents of the model are spin-1/2 local spins. The lattice is of honeycomb geometry.
        '''
        if width%2 != 0 or height%2 != 0:
            raise ValueError('Both width and height are required to be even to ensure a valid PBC')
        
        self.width = int(width)
        self.height = int(height)
        self.N = int(width*height) # total number of sites
        self.dim = int(2**(width*height)) # 2^N, the dimenIon of the total Hilbert space
        
        self._Jx = Jx
        self._Jy = Jy
        self._Jz = Jz

        self.I = sp.identity(2, format="csr")
        self.X = np.array([[0,1],[1,0]], dtype = np.complex64)
        self.Y = np.array([[0,-1j],[1j,0]], dtype = np.complex64)
        self.Z = np.array([[1,0],[0,-1]], dtype = np.complex64)

        self.rho = np.array([])

        return
    
    @property
    def Jx(self):
        return self._Jx
    @Jx.setter
    def Jx(self, Jx: float):
        self._Jx = Jx
        self.calculte_CorrTable()
        return
    
    @property
    def Jy(self):
        return self._Jy
    @Jy.setter
    def Jy(self, Jy: float):
        self._Jy = Jy
        self.calculte_CorrTable()
        return
    
    @property
    def Jz(self):
        return self._Jz
    @Jz.setter
    def Jz(self, Jz: float):
        self._Jz = Jz
        self.calculte_CorrTable()
        return
    
    @property
    def path(self):
        file_path = "w_%d_h_%d/Jx_%.3f_Jy_%.3f_Jz_%.3f" % (self.width, self.height, self.Jx, self.Jy, self.Jz)
        return file_path
    
    ################### correlation functions ###################
    """
    Here, the momentum is expressed in terms of reciprocal basis vectors
    b_1 = (2pi, 2pi/sqrt(3))
    b_2 = (0, 4pi/sqrt(3))
    as
    k = k1 b_1 + k2 b_2
    which are derived by the lattice basis vectors 
    a_1 = (1,0)
    a_2 = (1/2, sqrt(3)/2)
    """
    def sk(self, k1:float ,k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        return 2*np.sqrt(Jx**2 + Jy**2 + Jz**2 + 
                    2*Jx*Jy*np.cos(2*np.pi*k1) + 
                    2*Jx*Jz*np.cos(2*np.pi*k2) + 
                    2*Jy*Jz*np.cos(2*np.pi*(k1 - k2)))
    
    def Rk(self, k1:float, k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        return 2*(Jx*np.cos(2*np.pi*k2) + Jy*np.cos(2*np.pi*(k1 - k2)) + Jz)
    
    def Ik(self, k1:float, k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        return 2*(Jx*np.sin(2*np.pi*k2) - Jy*np.sin(2*np.pi*(k1 - k2)))
    
    def corr_init(self, na:int, nb:int)->float:
        """
        Two-points correlation function <-icc>. Notice the existance of the imaginary number -i
        This function is valid only for PBC
        """
        xa, ya, mua = self.dictionary_n_to_c(label=na) 
        xb, yb, mub = self.dictionary_n_to_c(label=nb)
        dx, dy = xb - xa, yb - ya 
        corr_fun = 0
        for nx in range(int(self.width/2)):
            for ny in range(self.height):
                k1 = nx / (self.width/2)
                k2 = ny / (self.width/2)
                R, I, s = self.Rk(k1=k1, k2=k2), self.Ik(k1=k1, k2=k2), self.sk(k1=k1, k2=k2)
                phase = 2*np.pi*(k1*dx + k2*dy)
                corr_fun += (2/self.N) * (R*np.cos(phase) + I*np.sin(phase)) / s
        return corr_fun
    
    def calculte_CorrTable(self):
        CorrTable = {}
        for n in range(self.N):
            if n%2 == 0:    # BB pairing is always zero
                continue

            CorrTable[n] = self.corr_init(na=n, nb=0)
        
        os.makedirs(name=self.path, exist_ok=True)
        with open(self.path  + '/' + "CorrTable.json", "w") as file:
            json.dump(CorrTable, file)  
        return
    
    def corr(self, na:int, nb:int, update:bool=False)->float:
        """
        Two-points correlation function <-icc>. Notice the existance of the imaginary number -i
        This function is valid only for PBC
        """
        if update:
            os.remove(self.path+'/'+'CorrTable.json')

        if os.path.isfile(self.path+'/'+'CorrTable.json'):
            with open(self.path + '/' + "CorrTable.json", "r") as file:
                CorrTable = json.load(file)

            la, sa = self.dictionary_n_to_l(label=na)
            lb, sb = self.dictionary_n_to_l(label=nb)

            dl, ds = (la - lb)%self.height, (sa - sb)%self.width
            dn = self.dictionary_l_to_n(layer=dl, site=ds)
            
            return CorrTable["%d" % dn] 
        else:
            self.calculte_CorrTable()
            return self.corr_init(na=na, nb=nb)
    
    def Coefficient(self, setA:list, setB:list)->float:
        '''
        Calculate the coefficient in front of each basis (configuration) by Wick's theorem.
        <(-ic_{a1} c_{b1}) (-ic_{a2} c_{b2})... > = sum over all possible parings of 
        <-ic_{ai} c_{bj}> products.
        The coefficients can be expressed as the determinents of correlation matrices defined as G_{ij} = <c_Ai c_Bj>
        The coefficient in front of each basis (configuration). 
        siteA and siteB are np.arrays, whose elements are end points of the configuration.

        Be careful that the output only contains the <cc> part, the imaginary coefficient arisis from
        icc and permutations of fermionic operators are not included.

        But we should be careful that <(-ic1c2)(-ic3c4)> changes sign when we permute the fermionic operators to <(-ic1c4)(-ic3c2)>.

        Note that the correlators are purely imaginary, so the calculation can be done under dtype=float 
        '''
        if np.size(setA) != np.size(setB):
            return 0
            #print('\033[31mError: end-point set sizes unmatched\033[0m')
        else:
            if np.size(setA) == 0:   # No end point means vacuum: identity
                return 1
            else:
                # ordering the labels of the lattice points
                setA.sort()
                setB.sort()

                coe = 0
                # determininant method
                corr_matrix = np.zeros([len(setA), len(setB)])
                for i in range(len(setA)):
                    for j in range(len(setB)):
                        corr_matrix[i,j] = self.corr(na=setA[i], nb=setB[j])   
                coe = np.linalg.det(corr_matrix)
                return coe
    
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
    
    def PauliString(self, pauliword:dict)->np.ndarray:
        '''
        Construct a corresponding Pauli string operator according to the given pauliword.

        ### Pauliword
        The format of a pauliword is like

        {
            sitelabel_1: type_1
            sitelabel_2: type_2
            ...
        }
        where each sitelabel_i is an integer and each type_i is 'X', 'Y', or 'Z'
        '''
        matrix = None
        for i in range(self.N):
            if i in pauliword:
                if pauliword[i] == 'X':
                    s = self.X
                elif pauliword[i] == 'Y':
                    s = self.Y
                else:   # pauliword[i] == 'Z'
                    s = self.Z
            else:
                s = self.I
            matrix = s if matrix is None else sp.kron(matrix,s, format='csr')
        
        return matrix
    
    ################### string operator ###################
    
    def dimer_direction(self, n1: int, n2:int)->str:
        """
        Input two nearest neighboring sites n1 and n2, and then return the direction of the dimer, which will be 'X', 'Y' or 'Z'.
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
        if l2%self.height == (l1-1)%self.height and s1%self.width == (s2-1)%self.width:
            return 'Z'
        elif s2%self.width == (s1 + 1)%self.width and l1%self.height == l2%self.height:
            return 'X'
        elif s2%self.width == (s1 - 1)%self.width and l1%self.height == l2%self.height:
            return 'Y'
        else:
            raise ValueError("Invalid points for forming a dimer")
    
    def dimer(self, n1: int, n2:int)->np.ndarray:
        """
        Input two nearest neighboring sites n1 and n2, and then return the dimer operator in the Kitaev model \dimer(n1,n2)
        """
        dir = self.dimer_direction(n1=n1, n2=n2)
        
        if dir == 'Z':
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.Z, op2=self.Z)
        elif dir == 'X':
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.X, op2=self.X)
        elif dir == 'Y':
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.Y, op2=self.Y)
        else:
            raise ValueError("Invalid points for forming a dimer")
    
    def string_op_pauliword(self, na:int, nb:int)->tuple[np.complex64, dict]:
        """
        Generate the Pauli *words* of a single string, whose endpoints are na (in sublattice A) and nb (in sublattice B)
        The output is (ImagCoe, pauliword). The phase factor ImagCoe stems from the commutation relations of the Pauli operators.
        
        ### Pauliword
        The format of a pauliword is like

        {
            sitelabel_1: type_1
            sitelabel_2: type_2
            ...
        }
        where each sitelabel_i is an integer and each type_i is 'X', 'Y', or 'Z'
        """
        pauliword = {}

        la, sa = self.dictionary_n_to_l(label=na)
        lb, sb = self.dictionary_n_to_l(label=nb)
        
        l, s = la, sa   # initialize the moving point. RULE: always starts from the type-A endpoint

        ImagCoe_tot = 1 # the imaginary factor due to the Pauli algebra
        
        pauli_type = None   # the type of the Pauli spin we want to add at the current site
        while l != lb or s != sb:

            ########### move the point to generate (l_next, s_next) ###########
            if lb == l%self.height:
                if sb > s%self.width:
                    l_next, s_next = l, s+1
                    #print('right')
                else:   #sb < s
                    l_next, s_next = l, s-1
                    #print('left')
            elif lb > l%self.height:
                if self.SubLatID_l(layer=l, site=s) == 0:   # sublattice B
                    l_next, s_next = l+1, s-1
                    #print('upper left')
                elif self.SubLatID_l(layer=l, site=s) == 1:   # sublattice A
                    l_next, s_next = l, s+1
                    #print('right')
            elif lb < l%self.height:
                if self.SubLatID_l(layer=l, site=s) == 0:   # sublattice B
                    l_next, s_next = l, s-1
                    #print('left')
                elif self.SubLatID_l(layer=l, site=s) == 1:   # sublattice A
                    l_next, s_next = l-1, s+1
                    #print('lower right')
            else:
                raise ValueError('invalid algorithm')
            ####################################################################
            
            # represent the points in terms of labeling numbers
            n, n_next = self.dictionary_l_to_n(layer=l, site=s), self.dictionary_l_to_n(layer=l_next, site=s_next)
            
            # determine the pauli type of the spin operator we want to add,
            if pauli_type is None:  # initial step
                dir = self.dimer_direction(n1=n, n2=n_next) # the direction of the current dimer
                #print('dir = ' + dir)
                pauli_type = dir    # at the first step, the Pauli spin at the initial point is in the same direction as the initial dimer, and it does not intersects with other Pauli spins
            else:
                dir_next = self.dimer_direction(n1=n, n2=n_next)
                #print('dir = ' + dir)
                #print('dir_next = ' + dir_next)
                ImagCoe, pauli_type = PauliMultRule(type1=dir, type2=dir_next) # equivalent to multiplying two dimer operators
                #print('pauli_type = ' + pauli_type)
                ImagCoe_tot = ImagCoe_tot * ImagCoe # update the imaginary coefficient
                dir = dir_next  # update the dimer direction
            
            #  collect the pauli type into the sets
            if pauli_type in ['X', 'Y', 'Z']:
                pauliword[n] = pauli_type
            else:
                raise ValueError('invalid algorithm for constructing string operators')
            
            l, s = l_next%self.height, s_next%self.width   # update

        # final step: for the terminal endpoint
        pauli_type = dir    # at the final step, the Pauli spin at the last point is in the same direction as the last dimer, and it does not intersects with other Pauli spins
        #  collect the pauli type into the sets
        if pauli_type in ['X', 'Y', 'Z']:
                pauliword[n] = pauli_type
        else:
            raise ValueError('invalid algorithm for constructing string operators')
            
        return ImagCoe_tot, pauliword
    
    def string_op(self, na:int, nb:int)->np.ndarray:
        """
        Generate the string operator of a single string, whose endpoints are na (in sublattice A) and nb (in sublattice B)
        """
        ImagCoe_tot, pauliword = self.string_op_pauliword(na=na, nb=nb)
        return  ImagCoe_tot * self.PauliString(pauliword=pauliword)
    
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

            #print(na, nb)
            Sigma = Sigma @ self.string_op(na=na, nb=nb)

            counting += 1

        return Sigma
    
    ################### constructing rho ###################
    
    def single_plaquette_pauliword(self, site:int)->dict:
        """
        Return the Pauli word of a plaquette operator whose lower left corner is located at ``site``, which must be of type-B and is labeled by an integer
        """
        PlaqPauliword = {}

        movepoint = site    # The starting point, which is by default the lower left corner vertex of the rectangle (hence type-B), and circling counterclockwise 
        l, s = self.dictionary_n_to_l(label=movepoint)

        PlaqPauliword[movepoint] = 'X'
        
        s += 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Z'
        
        s += 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Y'
        
        l += 1
        s -= 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'X'

        s -= 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Z'

        s -= 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Y'
        return  PlaqPauliword
    
    def plaquette_projector(self)->tuple[int, np.ndarray]:
        """
        The output is the number of plaquettes with the plaquette projector
        """

        ### sort lattice sites into A and B sublattices
        A_all = []
        B_all = []
        
        nA = 0
        nB = 0
        for i in np.arange(self.N, dtype=int):
            if i%2 == 0:
                B_all.append(i)
            else:
                A_all.append(i)

        PlaquetteProjector = sp.identity(self.dim, format='csr')    # initialize the plaquette projector
        
        p_number = 0
        
        for start in B_all:    # each starting point is the lower left corner vertex of the rectangle (hence type-B), and circling counterclockwise 
            PlaqPauliword = self.single_plaquette_pauliword(site=start)

            LocalPla = 0.5*(sp.identity(self.dim, format='csr') + self.PauliString(pauliword=PlaqPauliword))

            PlaquetteProjector = PlaquetteProjector @ LocalPla
            p_number += 1
        
        return p_number, PlaquetteProjector
    
    def wilson_loop(self, direction:str)->np.ndarray:
        """
        The variable direction is either 'horizontal' or 'vertical'. Inputs lying outside these two options are not allowed currently.
        """
        LoopPauliword = {}
        
        l, s = 0, 0   # moving point, which starts cricling the torus from the site n=0
        comeback = False
        if direction == 'horizontal':
            while not comeback:
                s += 1
                s = s%self.width    # PBC

                LoopPauliword[self.dictionary_l_to_n(layer=l, site=s)] = 'Z'
                
                if s%self.width == 0:
                    comeback = True
        elif direction == 'vertical':
            while not comeback:
                # moving to the upper left
                s -= 1
                l += 1
                s%self.width    # PBC
                l%self.height    # PBC

                LoopPauliword[self.dictionary_l_to_n(layer=l, site=s)] = 'Y'

                # moving to the right
                s += 1
                s%self.width    # PBC
                l%self.height    # PBC

                LoopPauliword[self.dictionary_l_to_n(layer=l, site=s)] = 'X'

                if s%self.width == 0:
                    comeback = True
        else:
            raise ValueError("The variable direction is either 'horizontal' or 'vertical'. Inputs lying outside these two options are not allowed currently.")
        
        return  self.PauliString(pauliword=LoopPauliword)
    
    def topo_projector(self)->np.ndarray:
        """
        0.25 * (I + Wh) @ (I + Wv) 
        """
        Wh, Wv = self.wilson_loop(direction='horizontal'), self.wilson_loop(direction='vertical')
        I = sp.identity(self.dim, format='csr')
        return  0.25 * (I + Wh) @ (I + Wv) 
    
    def EndpointSets(self)->list[list[list,list]]:
        """
        output all the possible sets of endpoints

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
                B_all.append(i)
            else:
                A_all.append(i)
        
        for n in range(np.size(A_all)+1):   # loop over number of pairings. n=0 means no pairing, i.e. vacuum: identity matrix. n=2 means 2 pairings, i.e. 4 end points.
            comb_A = np.array(list(combinations(A_all, n)))
            comb_B = np.array(list(combinations(B_all, n)))
            for i in range(comb(np.size(A_all), n, exact=True)):    # loop over all possible choices of n type-A sites
                for j in range(comb(np.size(B_all), n, exact=True)):    # loop over all possible choices of n type-B sites
                    pairing.append([comb_A[i,:], comb_B[j,:]])
        return pairing
    
    def calculate_coe(self, update:bool=False)->dict:
        """
        calculate the coefficients (expectation values of the string operators) and save the dictionary ``{'D': coe(D)}``
        """
        if update:
            os.remove(self.path  + '/' + "coefficients.json")

        if os.path.isfile(self.path  + '/' + "coefficients.json"):
            with open(self.path  + '/' + "coefficients.json", "r") as file:
                coe_dic = json.load(coe_dic, file)  
            return  coe_dic
        else:
            coe_dic = {}
            config = self.EndpointSets()

            for D in range(len(config)):
                setA, setB = config[D]

                coe = self.Coefficient(setA=setA, setB=setB)
                coe_dic[D] = coe

                #print('\033[34mProgress Report: Computing coefficients\033[0m')
                print('\033[34mProgress Report: The %d-th configuration (size: %d sites) done\033[0m' % (D, 2*len(setA)))

            with open(self.path  + '/' + "coefficients.json", "w") as file:
                    json.dump(coe_dic, file)  
            return coe_dic
    
    def density_matrix(self, update:bool=False)->np.ndarray:
        """
        Generate the density matrix (the entire system). 
        NOTE: the format of the output and saved density matrix is ``scipy.sparse``. To load the density matrix, please use ``sp.load_npz`` instead of ``np.load``
        """
        if update:  # recalculate the density matrix
            os.remove(self.path+'/'+'density-matrix.npz')

        if os.path.isfile(self.path+'/'+'density-matrix.npz'):
            return sp.load_npz(self.path+'/'+'density-matrix.npz')   
        else:
            config = self.EndpointSets()

            rho = None
            for D in range(len(config)):
                setA, setB = config[D]

                Sigma = self.SigmaOp(setA=setA, setB=setB)
                #print('\033[34mProgress Report: Constructing string operators\033[0m')

                coe = self.Coefficient(setA=setA, setB=setB)
                #print('\033[34mProgress Report: Computing coefficients\033[0m')

                rho = coe * Sigma if rho is None else rho + coe * Sigma
                print('\033[34mProgress Report: The %d-th configuration (size: %d sites) done\033[0m' % (D, 2*len(setA)))
            
            Np, p_proj = self.plaquette_projector()
            rho = 2 ** (-(self.N - Np)) * self.topo_projector() @  p_proj @ rho 
            self.rho = rho

            os.makedirs(name=self.path, exist_ok=True)
            sp.save_npz(self.path+'/'+'density-matrix.npz', rho)    # do not use np.save to save a sparse matrix
            
            return  rho

    ################### Translational Symmetry Trick ###################
    # Here, I would like to take advantage of translational symmetry and the nice property of the Pauli string.
    # Take the product spin states as the basis vectors, the value of <a|Sigma|b>, where |a> = 

    def state_to_spin(self, label:int)->list:
        """
        translate the n-th state to its spin configuration [s1, s2, ..., sN]
        # Convention
        0: up, 1: down

        0 -> [...,0,0,0]
        1 -> [...,0,0,1]
        2 -> [...,0,1,0]
        """
        bin_list = [int(x) for x in bin(label)[2:]]
        
        # fill in the missing spins (which are zeros)
        return [0 for n in range(int(self.N - len(bin_list)))] + bin_list
    
    def spin_to_state(self, config:list)->int:
        """
        translate a spin configuration [s1, s2, ..., sN] to its correaponding state label. 
        This is the inverse function of ``state_to_spin``
        # Convention
        0: up, 1: down

        [...,0,0,0] -> 0
        [...,0,0,1] -> 1
        [...,0,1,0] -> 2
        """
        return  int("".join(map(str, config)), 2)
    
    def translation_op(self, config:list, move:list[int,int])->list:
        """
        # Variables
        ``config``: the input spin configuration we want to translate
        ``move``: [x, y]: the vector of how far we want to translate (in the unit of unit cell, not lattice site)

        Define how a spin configuration are translated to another one
        Take a three-spin system with PBC for example, we have T([1,0,0]) = T([0,1,0])
        For a 4 by 4 lattice geometry, the labeling for the sites is
        [[12,   13, 14, 15]
        [8,     9,  10, 11]
        [4,     5,  6,  7]
        [0,     1,  2,  3]]
        RULE: left to right, lower to upper

        NOTE: The unit vector in the x-direction crosses TWO lattice sites
        """
        config_trans = config.copy()
        for l in range(self.height):
            for s in range(self.width):
                n = self.dictionary_l_to_n(layer=l, site=s) # the unit vector in the x-direction crosses TWO lattice sites
                config_trans[n] = config[self.dictionary_l_to_n(layer = l - move[1], site = s - 2*move[0])] # The factor 2 preceding ``move[0]`` is due to the fact that the unit vector in the x-direction crosses TWO lattice sites
        return  config_trans
    
    def pauli_string_act(self, state:int, pauliword:dict)->tuple[np.complex64, int]:
        """
        # Output:
        (phase, state_label)

        Compute 
        
        PauliOp|state> = phase |state'>

        Convention: 0: up, 1: down
        """
        config = self,self.state_to_spin(label=state)
        
        phase = 1   # Pauli Y and Z gives an additional phase factor
        for i in range(self.N): # here, i is the label of the lattice sites
            if pauliword[i] == 'X':
                config[i] = (config[i]+1)%2 # spin flipping
            elif pauliword[i] == 'Y':
                phase = phase * (-1)**config[i] * 1j    # phase
                config[i] = (config[i]+1)%2 # spin flipping
            elif pauliword[i] == 'Z':
                phase = phase * (-1)**config[i] # phase
            else:
                pass
        return  phase, self.spin_to_state(config=config)
    
    def pauli_string_act_vec(self, basis:list, state_vec:np.ndarray, pauliword:dict)->np.ndarray:
        """
        Similar to the function ```pauli_string_act```, but the the input and the output are vectors represented by a specific set of ```basis```

        ``basis``:list = [n1, n2,...], where each ni is the state label

        ``state_vec``: ``np.ndarray`` = (v1, v2, ...)^T

        output: ``np.ndarray`` = (v'1, v'2, ...)^T

        op (v1, v2, ...)^T = (v'1, v'2, ...)^T
        """
        # check if the size of state_vec matches that of the basis
        if len(basis) != len(state_vec):
            raise ValueError('Because state_vec is represented in terms of basis vectors, the size of the given basis set must be the same as the size of the input vector')
        
        output_vec = np.zeros_like(state_vec)

        for i in range(len(basis)):
            # opn is the label of the output basis: 
            # OP |basis[i]> = coe |opn> = coe |basis[f]>
            coe, opn = self.pauli_string_act(state=basis[i], pauliword=pauliword) 

            if opn in basis:
                f = basis.index(opn)    # OP |basis[i]> = coe |opn> = coe |basis[f]>
            else:
                raise ValueError('OP |basis[i]> lies outside the basis set. The basis should be closed under this operation.')
                
            # OP v[i]|basis[i]> = coe v[i] |opn> = coe v[i] |basis[f]>
            output_vec[f] += coe * state_vec[i]

        return  output_vec
    
    def pauli_string_sandwitch(self, state_1:int, state_2:int, pauliword:dict)->np.complex64:
        """
        # Variables
        ``state_1``, ``state_2``: the state labels of the spin configurations
        ``paulistr``: [PauliX, PauliY, PauliZ], each set contains the sites where Pauli spin operators of that direction are located
        
        # Function
        Compute <state_1|Pauli_String_Op|state_2>
        """
        config_1, config_2 = self.state_to_spin(label=state_1), self.state_to_spin(label=state_2)
        
        value = 1
        for i in range(self.N): # here, i is the label of the lattice sites
            if i in pauliword:
                if pauliword[i] == 'X':
                    S = self.X
                elif pauliword[i] == 'Y':
                    S = self.Y
                elif pauliword[i] == 'Z':
                    S = self.Z
            else:
                S = self.I

            value = value * S[config_1[i], config_2[i]]
        
        return  value
    
    def representative_set(self, update:bool=False)->list:
        
        if update:
            os.remove("w_%d_h_%d/representative_set.json" % (self.width, self.height))

        if os.path.isfile("w_%d_h_%d/representative_set.json" % (self.width, self.height)):
            with open("w_%d_h_%d/representative_set.json" % (self.width, self.height), "r") as file:
                rep_list = json.load(file)
        else:
            state_list = list(range(self.dim))
            rep_list = []

            find_next_rep = True
            while find_next_rep:
                rep_state = state_list[0]
                rep_list.append(rep_state) # start with the first element of the state list at each iteration step
                for x in range(int(self.width/2)): # the unit vector in the x-direction crosses TWO lattice sites
                    for y in range(self.height):
                        nt = self.spin_to_state(config=self.translation_op(config=self.state_to_spin(label=rep_state), move=[x,y]))
                        if nt in state_list:
                            #print('remove %d' % nt)
                            if nt == 65535:
                                print('the representative that is equivalent to 65535 is %d' % rep_state)
                            state_list.remove(nt)

                if state_list == []:
                    find_next_rep = False

            with open("w_%d_h_%d/representative_set.json" % (self.width, self.height), "w") as file:
                json.dump(rep_list, file)

        return  rep_list
    
    def period_dict(self, update:bool=False)->dict:
        """
        A period R is the smallest integer such that T^R |state> = |state>
        
        # Output:
        {
            "rep1": [Rx_1, Ry_1]
            "rep2": [Rx_2, Ry_2]
            ...
        }
        where each ``repi`` is an integer
        """
        if update:
            os.remove("w_%d_h_%d/period_dict.json" % (self.width, self.height))

        if os.path.isfile("w_%d_h_%d/period_dict.json" % (self.width, self.height)):
            with open("w_%d_h_%d/period_dict.json" % (self.width, self.height), "r") as file:
                period_dic = json.load(file)
        else:
            rep_list = self.representative_set()
            period_dic = {}
            for n in rep_list:
                ######## search the period in x
                x_search = True
                Rx = 1  # period R=0 trivially satisfies T^R = 1
                while x_search:
                    nt = self.spin_to_state(config=self.translation_op(config=self.state_to_spin(label=n), move=[Rx,0]))
                    if nt != n:
                        Rx += 1
                    else:
                        x_search = False
                ######## search the period in y
                y_search = True
                Ry = 1
                while y_search:
                    nt = self.spin_to_state(config=self.translation_op(config=self.state_to_spin(label=n), move=[0,Ry]))
                    if nt != n:
                        Ry += 1
                    else:
                        y_search = False
                period_dic["%d" % n] = [Rx, Ry]
            with open("w_%d_h_%d/period_dict.json" % (self.width, self.height), 'w') as file:
                json.dump(period_dic, file)
        return  period_dic
    
    def momentum_sectors(self, update:bool=False)->dict:
        """
        # Output:
        {
            "(m1_x,m1_y)": [rep1_k1, rep2_k1, ...]_k1, 
            "(m2_x,m2_y)": [rep1_k2, rep2_k2, ...]_k2,
            ... 
        }

        First we have to define the periodicity R of a state psi, which is the smallest integer such that T^R |psi> = |psi>
        An allowed representative in the momentum sector k must satisfy exp(ikR) = 1. Otherwise, the corresponding momentum state generated from such a representative will vanish.
        Each sector represents a diagonal block, with its corresponding momentum, of the density matrix
        
        The momentum states we consider here are
        m_mu = 0, 1, 2, ..., N_mu-1
        k_mu = m_mu * 2pi/N_mu
        """
        if update:
            os.remove("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height))

        if os.path.isfile("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height)):
            with open("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height), "r") as file:
                momentum_sectors_dict = json.load(file)
        else:
            rep_list = self.representative_set()
            period_dict = self.period_dict()

            momentum_sectors_dict = {}
            for mx in range(int(self.width/2)):  # the unit vector in the x-direction crosses TWO lattice sites
                for my in range(self.height):
                    
                    k_sector = []   #   start collecting representatives into the momentum sector
                    for n in rep_list:
                        [Rx, Ry] = period_dict["%d" % n]
                        #print("Rx=%d, Ry=%d, mx=%d, my=%d" % (Rx, Ry, mx, my))
                        if mx*Rx % (self.width/2) == 0 and my*Ry % self.height == 0:    # meaning that exp(ikR) = 1
                            k_sector.append(n)
                    momentum_sectors_dict["(%d,%d)" % (mx, my)] = k_sector
            with open("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height), "w") as file:
                json.dump(momentum_sectors_dict, file)
        
        return  momentum_sectors_dict

    def diagonal_block(self, basis:list, op:dict, **kwargs)->np.ndarray:
        """
        # Variables
        ## ``basis``: list
        The variable ``basis`` is a list of state labels

        basis = [n1, n2, ...]

        ## ``op``: dict
        Operators are expanded in terms of Pauli strings and are stored in the following form:
        {
            "component0": (coe_0, pauliword_0)
            "component1": (coe_1, pauliword_1)
            "component2": (coe_1, pauliword_2)
            ...
        }
        The mathmatical expression of the operator is then

        op = sum_{i} coe_i * PauliStr

        ## ``**kwargs``
        - hermitian
        - symmetric

        # Output:
        [block]_{n1,n2} = <n1| op |n2> = sum_{i} coe_i * <n1| PauliStr |n2>
        """
        ishermitian, issymmetric = False, False
        if 'hermitian' in kwargs:
            ishermitian = kwargs['hermitian']
        if 'symmetric' in kwargs:
            issymmetric = kwargs['symmetric']

        block_mat = np.zeros([len(basis), len(basis)], dtype=np.complex64)
        
        for cpnt in range(len(op)):    # the number of components of the operator expanded in terms of Pauli strings
            coe, pauliword = op['component%d'%cpnt]  # read the information of the current component
            
            # construct the matrix of the current component
            cpnt_mat = np.zeros_like(block_mat, dtype=np.complex64)
            for n1 in basis:
                for n2 in basis:
                    if ishermitian:
                        if n2 < n1:
                            cpnt_mat[n1,n2] = cpnt_mat[n2,n1].conjugate()
                            continue
                    elif issymmetric:
                        if n2 < n1:
                            cpnt_mat[n1,n2] = cpnt_mat[n2,n1]
                            continue
                    else:
                        pass
                    cpnt_mat[n1,n2] = coe * self.pauli_string_sandwitch(state_1=n1, 
                                                                         state_2=n2,
                                                                         pauliword=pauliword)
            # add the current component to the total matrix
            block_mat = block_mat + cpnt_mat
        return  block_mat
    
    def diagonal_block_LinearOperator(self, basis:list, op:dict)->LinearOperator:
        """
        This function does the same thing as the function ``diagonal_block``, but the output is a LinearOperator

        # Variables
        ## ``basis``: list
        The variable ``basis`` is a list of state labels

        basis = [n1, n2, ...]

        ## ``op``: dict
        Operators are expanded in terms of Pauli strings and are stored in the following form:
        {
            "component1": (coe_1, pauliword_1)
            "component2": (coe_2, pauliword_2)
            ...
        }
        The mathmatical expression of the operator is then

        op = sum_{i} coe_i * PauliStr

        ## ``**kwargs``
        - 'hermitian'
        - 'symmetric'

        # Output:
        [block]_{n1,n2} = <n1| op |n2> = sum_{i} coe_i * <n1| PauliStr |n2>
        """
        block_mat = None
        for cpnt in len(op):    # the number of components of the operator expanded in terms of Pauli strings
            coe, pauliword= op['component%d'%cpnt]  # read the information of the current component (cpnt)
            
            # define the matvec function of the current component
            # vec = v1*basis1 + v2*basis2 + ... = (v1, v2, ...)
            matvec_cpnt = lambda vec: coe * self.pauli_string_act_vec(basis=basis,
                                                                      state_vec=vec,
                                                                      pauliword=pauliword)
            block_mat = LinearOperator((len(basis), len(basis)), matvec=matvec_cpnt, dtype=np.complex64) if block_mat is None else block_mat + LinearOperator((len(basis), len(basis)), matvec=matvec_cpnt, dtype=np.complex64)

        return  block_mat
    
    def SigmaOp_pauliword(self, setA:list, setB:list)->tuple[np.complex64, dict]:
        """
        Producing the pauliword of the string operator of a general edge configuration, which may contain more than one string. setA consists of the type-A endpoints of the edge configuration, while setB contains the type-B ones. 
        
        SigmaOp = ImagCoe * PauliString = stringop1 @ stringop2 @ stringop3....
        """
        # A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints
        if len(setA) != len(setB):
            raise ValueError('A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints')
        
        # sort the sets such that the elements are arranged in an increasing order
        setA.sort()
        setB.sort()

        ImagCoe_tot = 1 # initialize the total phase factor ImagCoe
        SigmaPauliWord_tot = None   # initialize the pauliword of the output Pauli string operator

        counting = 0    # count for the setB
        for na in setA:
            nb = setB[counting]

            imcoe, stringpauliword = self.string_op_pauliword(na=na, nb=nb)

            ImagCoe_tot = ImagCoe_tot * imcoe
            SigmaPauliWord_tot = stringpauliword if SigmaPauliWord_tot is None else pauliwords_mult(pauliword1=SigmaPauliWord_tot, pauliword2=stringpauliword)

            counting += 1

        return ImagCoe_tot, SigmaPauliWord_tot
    
    def plaquette_projector_momentum_block(self, k:list[int,int], Z2sym:bool=True)->tuple[int, np.ndarray]:
        """
        The output is the number of plaquettes with the plaquette projector
        """

        ### sort lattice sites into A and B sublattices
        A_all = []
        B_all = []
        
        nA = 0
        nB = 0
        for i in np.arange(self.N, dtype=int):
            if i%2 == 0:
                B_all.append(i)
            else:
                A_all.append(i)

        k_basis_set = self.momentum_sectors()['(%d,%d)'%(k[0],k[1])]    # give the basis sector
        
        # initialize the plaquette projector, which is the identity of the momentum sector
        PlaquetteProjector = sp.identity(len(k_basis_set))    
        
        p_number = 0
        for start in B_all:    # each starting point is the lower left corner vertex of the rectangle (hence type-B), and circling counterclockwise 
            PlaqPauliword = self.single_plaquette_pauliword(site=start)

            LocalPla = 0.5*(sp.identity(self.dim, format='csr') + self.diagonal_block(basis=k_basis_set,
                                                                                      op={"component0":(1, PlaqPauliword)},
                                                                                      hermitian=True))

            PlaquetteProjector = PlaquetteProjector @ LocalPla
            p_number += 1
        
        return p_number, PlaquetteProjector
    
    def density_matrix_momentum_block(self, k:list[int,int], Z2sym:bool=True, update:bool=False)->np.ndarray:
        """
        ``k``: momentum = [kx, ky]

        ``Z2sym``: whether the system possesses a Z2 symmetry
        """

        isZ2_filename = '_Z2sym' if Z2sym else ''

        pathname = self.path+'/'+'density-matrix_momentum-block' + isZ2_filename 
        os.makedirs(name=pathname, exist_ok=True)

        save_filename = pathname + '/' + 'kx_%d_ky_%d' % (k[0], k[1])

        if update:  # recalculate the density matrix
            os.remove(save_filename +'.npy')

        if os.path.isfile(save_filename +'.npy'):
            return np.load(save_filename +'.npy')   
        else:
            k_basis_set = self.momentum_sectors()['(%d,%d)'%(k[0],k[1])]    # give the basis sector

            config = self.EndpointSets()

            rho = None
            for D in range(len(config)):    # looping all the possible configurations
                setA, setB = config[D]

                Sigma = self.SigmaOp(setA=setA, setB=setB)
                #print('\033[34mProgress Report: Constructing string operators\033[0m')

                coe = self.Coefficient(setA=setA, setB=setB)
                #print('\033[34mProgress Report: Computing coefficients\033[0m')

                rho = coe * Sigma if rho is None else rho + coe * Sigma
                print('\033[34mProgress Report: The %d-th configuration (size: %d sites) done\033[0m' % (D, 2*len(setA)))
            
            Np, p_proj = self.plaquette_projector()
            rho = 2 ** (-(self.N - Np)) * self.topo_projector() @  p_proj @ rho 
            self.rho = rho

            os.makedirs(name=self.path, exist_ok=True)
            sp.save_npz(self.path+'/'+'density-matrix.npz', rho)    # do not use np.save to save a sparse matrix
            
            return  rho
    
################################## extra functions ##################################

def PauliMultRule(type1:str, type2:str)->tuple[np.complex64, str]:
        '''
        RULE: the Pauli type must be capital
        \sigma^type1 \sigma^type2 = ImagCoe \sigma^ResultType
        '''
        rule = {
            'XY': (1j, 'Z'),    'YX': (-1j, 'Z'),
            'YZ': (1j, 'X'),    'ZY': (-1j, 'X'),
            'ZX': (1j, 'Y'),    'XZ': (-1j, 'Y'),
            'idX': (1, 'X'),    'Xid': (1, 'X'),
            'idY': (1, 'Y'),    'Yid': (1, 'Y'),
            'idZ': (1, 'Z'),    'Zid': (1, 'Z'),
            'XX': (1, 'id'),
            'YY': (1, 'id'),
            'ZZ': (1, 'id'),
            'idid': (1, 'id')
            }
        comb_str = type1 + type2
        return  rule[comb_str]
        
def pauliwords_mult(pauliword1:dict, pauliword2:dict)->tuple[np.complex64, dict]:
    """
    Multiply two Pauli strings: 
    
    pualistr1*paulistr2 = ImagCoe * paulistr_output

    The output is the Pauli word corresponding to the resulting Pauli string ``paulistr_output``
    """
    outputPauliword = {}
    ImagCoe_total = 1
    union = pauliword1 | pauliword2
    for site in union:
        if site in pauliword1 and site in pauliword2:
            imcoe, new_type = PauliMultRule(type1=pauliword1[site], type2=pauliword2[site])
            ImagCoe_total = ImagCoe_total * imcoe
            outputPauliword[site] = new_type
        else:
            outputPauliword[site] = union[site]
    return  ImagCoe_total, outputPauliword

def read_num(input:str, prompt:str)->int:
    """
    Find the number after the prompt

    input = 'prompt num'
    """
    num_list = ''

    searching = True
    l = 0
    while searching:
        tobetest = ''
        for k in range(len(prompt)):
            tobetest += input[l+k]
            num_start_point = l+k+1 # the place of the first digit

        if tobetest == prompt:
            finddigit = True
            cursor = 0
            while finddigit:
                if num_start_point + cursor + 1 > len(input):
                    finddigit = False   # terminate the searching 
                    searching = False
                    break

                if input[num_start_point + cursor].isdigit():
                    num_list += input[num_start_point + cursor]
                    cursor += 1
                else:
                    finddigit = False   # terminate the searching 
                    searching = False
        l += 1
    return  int(num_list)