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

    def clean_disk(self):
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
                    data_chunk = np.asarray(list(f.read()))
                    user_data[j, i] = data_chunk
            for k in range(self.N, self.D):
                chunk_path = os.path.join(Config.DISK_PATH, "Disk" + str(k), "chunk" + str(k) + str(i))
                with open(chunk_path, 'rb') as f:
                    chunk = np.asarray(list(f.read()))
                    checksum[k-self.N, i] = chunk

        # generate new checksum
        new_checksum = self.compute_parity(user_data)

        # compare new checksum and old checksum
        corrupted_chunk = []
        for i in range(self.strip):
            failed_chunk = [] # store the failed chunk for i-th strip
            for j in range(self.M):
                r = (new_checksum[j, i, :] == checksum[j, i, :])
                if False in r:
                    print("Chunk" + str(j+self.N) + str(i) + " Corrupted!")
                    failed_chunk += ["Chunk" + str(j+self.N) + str(i)]
            # Only store the failed chunk
            if len(failed_chunk) > 0:
                corrupted_chunk += [failed_chunk]

        return corrupted_chunk

    def file_update(self):
        """
        Update the file in the RAID6 system
        """
        original_file = self.read_disk_data(len(self.input_file))
        temp_path = os.path.join(Config.DISK_PATH, "temp.txt")
        with open(temp_path, 'wb') as f:
            chunk = bytes(original_file)
            f.write(chunk)
        print("Please update the file " + temp_path + " ...")
        input("Press enter if file update is finished ...")

        self.clean_disk()
        self.read_file(temp_path)
        self.encode_data(self.input_file)


if __name__ == "__main__":
    raid6 = RAID6()
    raid6.clean_disk()
    raid6.read_file("data/sample.txt")
    raid6.encode_data(raid6.input_file)
    # data = raid6.read_disk_data(len(raid6.input_file))
    # print(bytes(data))
    # raid6.check_disk_corruption()
    # raid6.check_disk_exit()
    raid6.check_disk_corruption()
    # raid6.file_update()

