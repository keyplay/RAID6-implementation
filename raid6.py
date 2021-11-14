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
        self.gf = GaloisField(num_data_disk=self.N, num_check_disk=self.M)

        print("Number of Data Disk: " + str(self.N))
        print("Number of Checksum Disk: " + str(self.M))
        print("Chunk Size: " + str(self.chunk_size) + " Bytes")
        print("==============================")

        for i in range(self.D):
            directory = "Disk" + str(i)
            path = os.path.join(Config.DISK_PATH, directory)
            if not os.path.exists(path):
                os.mkdir(path)

    def clean_disk(self):
        """
        Clean the RAID6 disk
        """
        for d in os.listdir(Config.DISK_PATH):
            if "Disk" == d[:4]:
                for f in os.listdir(os.path.join(Config.DISK_PATH, d)):
                    os.remove(os.path.join(Config.DISK_PATH, d, f))

    def read_file(self, file):
        with open(file, 'rb') as f:
            self.input_file = list(f.read())

    def compute_parity(self, data):
        """
        Compute parity with Reed-Solomon Coding
            gf.vander : Vandermonde matrix, (M, N)
            data : 3-d matrix, (N, stripe_count, chunk_size)
        @return res : 3-d matrix, (M, stripe_count, chunk_size)
        """
        F = self.gf.vander
        D = data
        res = np.zeros([F.shape[0], D.shape[1], D.shape[2]], dtype=int)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                for k in range(res.shape[2]):
                    res[i][j][k] = self.gf.dot(F[i, :], D[:, j, k])

        return res

    def encode_data(self, data):
        """
        Store data into RAID6 file system
        """
        # determine stripe count and pad 0 to the end of the file
        stripe_size = self.chunk_size * self.N
        stripe_count = len(data) // stripe_size + 1
        self.strip = stripe_count
        stripe_size_byte = stripe_size * stripe_count
        padding_size = stripe_size_byte - len(data)
        padded_data = np.asarray(data + [0] * padding_size)

        # convert data into 3-d matrix, (N, stripe_count, chunk_size)
        padded_data = padded_data.reshape((stripe_count, self.N, self.chunk_size))
        padded_data = np.transpose(padded_data, (1, 0, 2))

        # calculate parity (checksum): FD = C
        parity = self.compute_parity(padded_data)
        data_with_parity = np.concatenate((padded_data, parity), axis=0)

        # write data into RAID6 disk
        for i in range(stripe_count):
            for j in range(self.D):
                directory = "Disk" + str(j)
                file = os.path.join(Config.DISK_PATH, directory, "chunk" + str(j) + str(i))
                with open(file, 'wb') as f:
                    chunk = bytes(data_with_parity[j, i, :].tolist())
                    f.write(chunk)
        print("Write successful")

    def read_disk_data(self, data_length):
        """
        Read data from RAID6 file system
        :param data_length: the data length for remove padded 0s
        :return: read data
        """
        content = []
        for i in range(self.strip):
            for j in range(self.N):
                chunk_path = os.path.join(Config.DISK_PATH, "Disk" + str(j), "chunk" + str(j) + str(i))
                with open(chunk_path, 'rb') as f:
                    content += list(f.read())

        content = content[:data_length]
        return content


if __name__ == "__main__":
    raid6 = RAID6()
    raid6.clean_disk()
    raid6.read_file("data/sample.txt")
    raid6.encode_data(raid6.input_file)
    data = raid6.read_disk_data(len(raid6.input_file))
    print(bytes(data))


