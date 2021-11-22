# RAID6-implementation

## Code
The code is implemented with Python3.7

Extra package:
- numpy

## File Introduction
#### config.py
This file configures the RAID6 system.
The configurations include:
1. the number of data disk
2. the number of checksum disk
3. the Galois Field degree
4. chunk size
5. the path towards the disk

#### ffield.py
This file implementes the mathmatical operation in the Galois Field, which includes:
1. addition
2. subtraction
3. multiplication
4. division
5. matrix multiplication
6. matrix inverse

#### raid6.py
This file implementing the controller of the RAID6 system, which support:
1. disks initialization
2. disks cleanup
3. read real file and write into the RAID6 file system
4. check the existence of disks
5. failed disk rebuild
6. check strip corruption
7. corrupted strip data rebuild
8. file update by user

## Running
running the RAID6 system by using command:

`> python raid6.py`
