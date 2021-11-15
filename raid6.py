"""
This file implementing RAID6 system read, store, and fault detection
"""
from config import Config
from ffield import GaloisField
import os
import numpy as np


class RAID6:
    """
    This class implementing the controller of the RAID6 system, including
        1. read file and store in the RAID6 file system
        2. read file from RAID file system
        3. detect disk loss
        4. detect disk corruption

        Params:
            self.N : number of data disk
            self.M : number of checksum disk
            self.D : number of disk, which is a sum of data disk and checksum disk
            self.chunk_size : number of bytes per chunk
            self.layer : number of chunks per disk, define the capacity of every disk
    """

    def __init__(self):
        """
        Initialize disk parameters and initialize the empty disk
        """
        self.N = Config.NUM_DATA_DISK
        self.M = Config.NUM_CHECKSUM_DISK
        self.D = self.N + self.M
        self.chunk_size = Config.CHUNK_SIZE
        self.layer = Config.DISK_LAYER
        self.input_file = None
        self.strip = 0
        self.data_disk_list  = list(range(self.N))
        self.check_disk_list = list(range(self.N, self.N+self.M))
        self.gf = GaloisField(num_data_disk=self.N, num_check_disk=self.M)
        
        self.init_disk()
        
        print("Number of Data Disk: " + str(self.N))
        print("Number of Checksum Disk: " + str(self.M))
        print("Chunk Size: " + str(self.chunk_size) + " Bytes")
        print("==============================")

    def init_disk(self):
        """
        Initialize disk of the RAID6 file system
        """
        for i in range(self.D):
            directory = "Disk" + str(i)
            path = os.path.join(Config.DISK_PATH, directory)
            if not os.path.exists(path):
                os.mkdir(path)

    def clean_disk(self, clean_list):
        """
        erase selected disks
        """
        for i in clean_list:
            d = "Disk{}".format(i)
            for f in os.listdir(os.path.join(Config.DISK_PATH, d)):
                os.remove(os.path.join(Config.DISK_PATH, d, f))
            print("\ndisk_{} is erased".format(i))

    def clean_all_disk(self):
        """
        Clean the RAID6 disk
        """
        for i in range(self.D):
            directory = "Disk" + str(i)
            path = os.path.join(Config.DISK_PATH, directory)
            if os.path.exists(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))

    def check_disk_exit(self):
        """
        check whether any disk lost
        :return: the lost disk id, return [] if no disk lost
        """
        lost_disk = []
        for i in range(self.D):
            directory = "Disk" + str(i)
            path = os.path.join(Config.DISK_PATH, directory)
            if not os.path.exists(path):
                lost_disk += [i]
                print(directory + " is lost!")

        return lost_disk

    def read_file(self, file):
        with open(file, 'rb') as f:
            self.input_file = list(f.read())

    def compute_parity(self, padded_data):
        """
        Compute parity with Reed-Solomon Coding
            gf.vander : Vandermonde matrix, (M, N)
            data : 3-d matrix, (N, stripe_count, chunk_size)
        @return res : 3-d matrix, (M, stripe_count, chunk_size)
        """
        vm = self.gf.vander
        res = np.zeros([vm.shape[0], padded_data.shape[1], padded_data.shape[2]], dtype=int)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                for k in range(res.shape[2]):
                    res[i][j][k] = self.gf.dot(vm[i, :], padded_data[:, j, k])

        return res

    def pad_data(self, input_data):
        """
        Pad 0s to the end of the input data
        :param input_data: original data
        """
        # determine stripe count and pad 0 to the end of the file
        stripe_size = self.chunk_size * self.N
        stripe_count = len(input_data) // stripe_size + 1
        self.strip = stripe_count
        stripe_size_byte = stripe_size * stripe_count
        padding_size = stripe_size_byte - len(input_data)
        padded_data = np.asarray(input_data + [0] * padding_size)

        # convert data into 3-d matrix, (N, stripe_count, chunk_size)
        padded_data = padded_data.reshape((stripe_count, self.N, self.chunk_size))
        padded_data = np.transpose(padded_data, (1, 0, 2))
        
        return padded_data

    def write_to_disk(self, data_with_parity):
        """
        Write data to RAID6 disk
        """
        for i in range(self.strip):
            for j in range(self.D):
                directory = "Disk" + str(j)
                file = os.path.join(Config.DISK_PATH, directory, "chunk" + str(j) + str(i))
                with open(file, 'wb') as f:
                    chunk = bytes(data_with_parity[j, i, :].tolist())
                    f.write(chunk)
        print("Write successful")

    def encode_data(self, input_data):
        """
        Store data into RAID6 file system
        """
        padded_data = self.pad_data(input_data)

        # calculate parity (checksum): FD = C
        parity = self.compute_parity(padded_data)
        data_with_parity = np.concatenate((padded_data, parity), axis=0)

        self.write_to_disk(data_with_parity)

    def read_chunk_data(self, stripe_num, chunk_num):
        """
        Read data from specific chunk
        :param stripe_num: chunk is in which stripe
        :param chunk_num: read data from which chunk
        :return: read data
        """
        chunk_path = os.path.join(Config.DISK_PATH, "Disk" + str(chunk_num), "chunk" + str(chunk_num) + str(stripe_num))
        with open(chunk_path, 'rb') as f:
            content = list(f.read())
        return content

    def read_disk_data(self, data_length):
        """
        Read data from RAID6 file system
        :param data_length: the data length for remove padded 0s
        :return: read data
        """
        content = []
        for i in range(self.strip):
            for j in range(self.N):
                content += self.read_chunk_data(i, j)

        content = content[:data_length]
        return content

    def check_disk_corruption(self):
        """
        Check whether any of the checksum disk is corrupted
        """
        user_data = np.zeros((self.N, self.strip, self.chunk_size), dtype=int)
        checksum = np.zeros((self.M, self.strip, self.chunk_size), dtype=int)

        # load data from RAID6 file system
        for i in range(self.strip):
            for j in range(self.N):
                chunk_path = os.path.join(Config.DISK_PATH, "Disk" + str(j), "chunk" + str(j) + str(i))
                with open(chunk_path, 'rb') as f:
                    user_data[j, i, :] = np.asarray(list(f.read()))
            for k in range(self.N, self.D):
                chunk_path = os.path.join(Config.DISK_PATH, "Disk" + str(k), "chunk" + str(k) + str(i))
                with open(chunk_path, 'rb') as f:
                    checksum[k, i, :] = np.asarray(list(f.read()))

        # generate new checksum
        new_checksum = self.compute_parity(user_data)

        # compare new checksum and old checksum
        corrupted_chunk = []
        for i in range(self.strip):
            for j in range(self.M):
                r = (new_checksum[j, i, :] == checksum[j, i, :])
                if False in r:
                    print("Chunk" + str(j+self.N) + str(i) + " Corrupted!")
                    corrupted_chunk += ["Chunk" + str(j+self.N) + str(i)]

        return corrupted_chunk    

    def rebuild_stripe_data(self, stripe_num, chunk_list):
        '''rebuild data from corrupted chunk
        :param stripe_num: corrupted chunk in which stripe 
        :param chunk_list: corrupted chunk
        :return: None
        '''
        
        #input("\nPress Enter to rebuild lost data ...\n")

        if len(chunk_list) > self.D:
            print("failed to rebuild data")
            return -1

        left_data = []
        left_parity = []
        left_data_chunk = list(set(self.data_disk_list) - set(chunk_list))
        left_check_chunk = list(set(self.check_disk_list) - set(chunk_list))

        for i in left_data_chunk:
            left_data.append(self.read_chunk_data(stripe_num, i))
        for j in left_check_chunk:
            left_parity.append(self.read_chunk_data(stripe_num, j))

        A = np.concatenate([np.eye(self.N, dtype=int), self.gf.vander], axis=0)
        A_= np.delete(A, obj=chunk_list, axis=0)

        E_= np.concatenate([np.asarray(left_data), np.asarray(left_parity)], axis=0)

        D = self.gf.matmul(self.gf.inverse(A_), E_)
        C = self.gf.matmul(self.gf.vander, D)

        E = np.concatenate([D, C], axis=0)

        for i in chunk_list:
            directory = "Disk" + str(i)
            file = os.path.join(Config.DISK_PATH, directory, "chunk" + str(i) + str(stripe_num))
            with open(file, 'wb') as f:
                to_write = bytes(E[i,:].tolist())
                f.write(to_write)
        
        print("rebuild data successfully\n")
        
    def rebuild_disk_data(self, disk_list):
        '''rebuild data from corrupted disk
        :param disk_list: corrupted disk
        :return: None
        '''
        if len(disk_list) > self.D:
            print("failed to rebuild data")
            return -1
        
        for stripe_num in range(self.N):
            self.rebuild_stripe_data(stripe_num, disk_list)
            
            
if __name__ == "__main__":
    raid6 = RAID6()
    raid6.clean_all_disk()
    raid6.read_file("data/sample.txt")
    raid6.encode_data(raid6.input_file)
    data = raid6.read_disk_data(len(raid6.input_file))
    print(bytes(data))
    
    clean_list = [0]
    raid6.clean_disk(clean_list)
    raid6.rebuild_disk_data(clean_list)
    
    rebuild_data = raid6.read_disk_data(len(raid6.input_file))
    print(bytes(data))

    print('Rebuild flag:', rebuild_data == data)

    input("\nPress Enter after delect data ...\n")
    #corrupted_chunk = raid6.check_disk_corruption()  # bug exists
    #raid6.rebuild_stripe_data(0, clean_list)