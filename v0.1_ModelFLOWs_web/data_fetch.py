import hdf5storage
from sys import platform

def fetch_data(path0, selected_file):
    # Check the operating system
    if platform in ["linux", "linux2"]:
        # linux
        Tensor_ = hdf5storage.loadmat(f'{path0}/{selected_file}')
        Tensor = list(Tensor_.values())[-1]
    elif platform == "darwin":
        # OS X
        Tensor_ = hdf5storage.loadmat(f'{path0}/{selected_file}')
        Tensor = list(Tensor_.values())[-1]
    elif platform in ["win32", "win64"]:
        # Windows
        Tensor_ = hdf5storage.loadmat(f'{path0}\{selected_file}')
        Tensor = list(Tensor_.values())[-1]
    return Tensor