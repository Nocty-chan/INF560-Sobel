INSTALL_PREFIX=/users/profs/2016/francois.trahay/soft/install

# HWLOC stuff
HWLOC_ROOT=$INSTALL_PREFIX/hwloc-1.11.5
PATH=$PATH:$HWLOC_ROOT/bin/
MANPATH=$MANPATH:$HWLOC_ROOT/share/man
LD_LIBRARY_PATH=$HWLOC_ROOT/lib:$LD_LIBRARY_PATH
PKG_CONFIG_PATH=$HWLOC_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH

# CUDA stuff
CUDA_ROOT=/usr/local/cuda
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_ROOT}/lib/
PATH=$PATH:${CUDA_ROOT}/bin/:${CUDA_ROOT}/open64/bin/

# MPI stuff
MPI_ROOT=$INSTALL_PREFIX/openmpi-2.0.1/
PATH=${MPI_ROOT}/bin:$PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MPI_ROOT}/lib
MANPATH=$MANPATH:$MPI_ROOT/share/man

# EZTrace
EZTRACE_ROOT=$INSTALL_PREFIX/eztrace-1.1-5/
PATH=$PATH:$EZTRACE_ROOT/bin

# Vampirtrace
VT_ROOT=$INSTALL_PREFIX/vampirtrace-5.14.4/
PATH=$PATH:$VT_ROOT/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VT_ROOT/lib

# ViTE
VITE_ROOT=$INSTALL_PREFIX/VITE-1.2.0-Linux
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VITE_ROOT/lib
PATH=$PATH:$VITE_ROOT/bin

export LD_LIBRARY_PATH
export PATH
export MANPATH
export PKG_CONFIG_PATH
