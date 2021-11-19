import numpy as np 

class GaloisField(object):
    def __init__(self, num_data_disk, num_check_disk, w = 8, modulus = 0b100011101): 
        '''inital setting
        field: GF(2^w)
        primitive polynomial: x^8+x^4+x^3+x^2+1, modulus is the coefficients of polynomial of x
        '''
        self.num_data_disk = num_data_disk
        self.num_check_disk = num_check_disk
        self.w = w
        self.modulus = modulus
        self.x_to_w = 1 << w 
        
        self.gflog = np.zeros((self.x_to_w, ), dtype=int)
        self.gfilog = np.zeros((self.x_to_w, ), dtype=int)
        self.vander = np.zeros((self.num_check_disk, self.num_data_disk), dtype=int)
        
        self.setup_tables()
        self.setup_vander()
       
    
    def setup_tables(self):
        b = 1
        for log in range(self.x_to_w - 1):
            self.gflog[b] = log
            self.gfilog[log] = b
            b = b << 1
            if b & self.x_to_w:
                b = b ^ self.modulus
    
    def setup_vander(self):
        for i in range(self.num_check_disk):
            for j in range(self.num_data_disk):
                self.vander[i][j] = self.power(j+1, i)
    
    def add(self, a, b):
        '''add is xor
        '''
        return a ^ b

    def sub(self, a, b):
        '''Subtraction is the same as add
        '''
        return self.add(a, b)
    
    def mult(self, a, b):
        '''Multiplication 
        :param a: multiplicand
        :param b: multiplier
        '''
        sum_log = 0
        if a == 0 or b == 0:
            return 0 
        sum_log = self.gflog[a] + self.gflog[b]
        if sum_log >= self.x_to_w - 1:
            sum_log -= self.x_to_w -1
        return self.gfilog[sum_log]
    
    def div(self, a, b):
        '''Division
        :param a: dividend
        :param b: divisor
        '''
        diff_log = 0
        if a == 0:
            return 0
        if b == 0:
            return -1
        diff_log = self.gflog[a] - self.gflog[b]
        if diff_log < 0:
            diff_log += self.x_to_w - 1
        return self.gfilog[diff_log]
    
    def power(self, a, n):
        '''Exponentiation
        :param a: base
        :param n: exponent
        '''
        n %= self.x_to_w - 1
        res = 1
        for i in range(n):
            res = self.mult(a, res)           
        return res
    
    def dot(self, v1, v2):
        '''Inner product of vector
        '''
        res = 0
        if len(v1) == len(v2):
            for i in range(len(v1)):
                res = self.add(res, self.mult(v1[i], v2[i]))
        return res

    def matmul(self, m1, m2):
        '''Matrix multiplication
        '''
        if m1.shape[1] == m2.shape[0]:
            res = np.zeros([m1.shape[0], m2.shape[1]], dtype=int)
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    res[i][j] = self.dot(m1[i, :], m2[:, j])
        else:
            print('error: dim of m1 and m2 is not consistent!')
            res = None
        return res

    def inverse(self, A):
        """
        cal the left inverse matrix of A
        :param A: matrix
        """
        if A.shape[0] != A.shape[1]:
            A_T = np.transpose(A)
            A_ = self.matmul(A_T, A)
        else:
            A_ = A
        A_ = np.concatenate((A_, np.eye(A_.shape[0], dtype=int)), axis=1)
        dim = A_.shape[0]
        for i in range(dim):
            if not A_[i, i]:
                for k in range(i + 1, dim):
                    if A_[k, i]:
                        break
                A_[i, :] = list(map(self.add, A_[i, :], A_[k, :]))
            A_[i, :] = list(map(self.div, A_[i, :], [A_[i, i]] * len(A_[i, :])))
            for j in range(i+1, dim):
                A_[j, :] = self.add(A_[j,:], list(map(self.mult, A_[i, :], [self.div(A_[j, i], A_[i, i])] * len(A_[i, :]))))
        for i in reversed(range(dim)):
            for j in range(i):
                A_[j, :] = self.add(A_[j, :], list(map(self.mult, A_[i, :], [A_[j,i]] * len(A_[i,:]))))
        A_inverse = A_[:,dim:2*dim]
        if A.shape[0] != A.shape[1]:
            A_inverse = self.matmul(A_inverse, A_T)

        return A_inverse
        
