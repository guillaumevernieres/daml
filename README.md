# DAML
Data Assimilation and Machine Learning integration framework

## Dependencies
This project assumes that the below dependencies have already been built.

#### spack-stack 1.5.1
On HPC, just load the GDASApp modules

#### JEDI
Just build your favorite jedi-bundle. This project makes use of
oops, ufo, ioda, vader, saber, soca, atlas, eckit, ...

####  Torch/PyTorch
Cloning and building pytorch

```
git clone --recursive --branch v2.3.1 https://github.com/pytorch/pytorch
cd pytorch
mkdir build
cd build
cmake -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DUSE_NCCL=OFF -DBUILD_CAFFE2_OPS=OFF -DUSE_MKLDNN=OFF -DUSE_DISTRIBUTED=ON -DCMAKE_INSTALL_PREFIX=<path to install> ..
make -j<n>
make install
```

## Applications

#### Bias/Error correction for MV based sea surface salinity retrievals
Coming soon ...

#### Unbalanced background error for sea surface height
Estimate the unblanced part of the background error for sea surface height.
Naive implementation that demonstrates the use of the JEDI and Torch libraries.

## Build process
For now, you will have to specify the root instalation of the JEDI repositories and Torch:

```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install \
      -DTorch_ROOT=<path to pytorch install> \
      -Doops_ROOT=<path to oops install> \
      -Datlas_ROOT=<path to atlas install> \
      ..
make -j<n>
make install
```
