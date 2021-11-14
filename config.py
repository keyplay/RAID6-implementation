class Config:
    """
    This class initialize the RAID system config:
        1. define data disk and checksum disk
        2. define chunk size (number of bytes in one chunk)
        3. define disk capacity (number of bytes in one disk)
    """
    NUM_DATA_DISK = 6
    NUM_CHECKSUM_DISK = 2
    CHUNK_SIZE = 16
    DISK_CAPACITY = 1024
    DISK_LAYER = DISK_CAPACITY // CHUNK_SIZE
    DISK_PATH = "data"




