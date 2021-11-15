"""
This file aims to provide necessary functions for RAID6 file read and store,
"""
import os
import numpy as np
import bitarray
import config


def init_disk():
    """
    Initialize disk based on config file
    """
    parent_dir = "data"
    for i in range(config.N):
        directory = "Disk" + str(i)
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)


if __name__ == "__main__":
    init_disk()
