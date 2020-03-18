# 1. Makefile.config 详解
Makefile文件
```C
Makefile.config.example## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
# 是否使用cudnn，默认不使用
# USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# 是否仅支持cup，注释后会编译gpu部分，因人而异，我自己学习支持cpu就好
CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# 选择支持库，可以先按默认都不选，包括下面对opencv，leveldb，lmdb库的选项都可以注释
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0
# This code is taken from https://github.com/sh1r0/caffe-android-lib
# USE_HDF5 := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# 可选g++版本，如果编译报错因g++库问题，可以尝试换个g++版本，比如曾经报错然后试过g++-5.x，g++-4.x等
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
# cuda安装命令，其实使用NVADIA gpu才会用到cuda,但是可以使用sudo apt-get install nvidia-cuda-toolkit
# 安装后，which cuda，添加路径编译
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
# For CUDA >= 9.0, comment the *_20 and *_21 lines for compatibility.
# 这个根据cuda版本不同选择注释，看上面注释，我是cuda7.5，所以注释*_60等
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
# BLAS库，安装atlas就使用这个吧
BLAS := atlas
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
# 这个是python库路径，可通过ubuntu的python库在 /usr/lib/x86-xxxx(忘了全拼，自己按tab补全吧)/python2.7/里面
PYTHON_INCLUDE := /usr/include/python2.7 \
		/usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
# anaconda是一个封装的环境，里面有编译caffe所需的库，如果通过anaconda安装caffe时候打开配置，并
# 注释上面python2.7的配置，修改成anaconda的环境路径，这里我不需要
# ANACONDA_HOME := $(HOME)/anaconda
# PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		# $(ANACONDA_HOME)/include/python2.7 \
		# $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include

# Uncomment to use Python 3 (default is Python 2)
# # python3的路径配置，我是2，所以不需要，根据实际情况配置
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
# 按默认的，如果使用anaconda，则换成下面的
PYTHON_LIB := /usr/lib
# PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# Python头文件路径，根据实际情况填写
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
# 如果想要caffe支持python接口，则这个打开，我会编译make && make pycaffe，所以需要，如果不编译，则
# 使用C++接口
# WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
# 库的配置，这里添加hdf的库，我环境会编译报错：can not find hdf5.h
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# NCCL acceleration switch (uncomment to build with NCCL)
# https://github.com/NVIDIA/nccl (last tested version: v1.2.3-1+cuda8.0)
# gpu相关
# USE_NCCL := 1

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# opencv支持，暂时不需要
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @
```

# 2. 使用anaconda和CMakeList来进行安装

1. 配置生成一个名为caffe的envs
```shell
create -n caffe python=2.7 或者3.6
source activate caffe        #进入caffe这个envs
```
2. 安装caffe的第三方依赖：
**atlas**、 **mkl**、 **openblas** 库
```shell
conda install libprotobuf=2.5 leveldb=1.19 snappy libopencv hdf5 boost numpy gflags=2.1.2 glog=0.3.4 lmdb atlas mkl openblas
```
选装CUDA，cudnn，其实anaconda也可以安装cuda-toolkit和cudnn，nccl库，但是由于安装的库并不直接包含nvcc，所以还是直接使用最传统的cuda安装方法，直接将cuda安装到系统目录/usr/local/cuda：
请参考各类教程，[如该教程](https://blog.csdn.net/wanzhen4330/article/details/81699769)

1. 针对caffe目录下的Cmakelist.txt，我们需要指定第三方库的位置.下面是经过测试的Cmakelist.txt，
```C
cmake_minimum_required(VERSION 2.8.7)
cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)
include(GNUInstallDirs)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" ON) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "3" CACHE STRING "Specify which Python version to use")

caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)



# This code is taken from https://github.com/sh1r0/caffe-android-lib
caffe_option(USE_HDF5 "Build with hdf5" ON)

#重点：：：：：：：请在这里把自己的anaconda的envs——caffe路径设置在这里，下面是我的路径，请务必更改
set(ENVS_INCLUDE    /home/archlab/anaconda3/envs/caffe_35/include)
set(ENVS_LIB        /home/archlab/anaconda3/envs/caffe_35/lib)
set(ENVS_EXECUTABLE /home/archlab/anaconda3/envs/caffe_35/bin)
#3rdparty path
#GLOG
set(GLOG_INCLUDE_DIR ${ENVS_INCLUDE} )
set(GLOG_LIBRARY ${ENVS_LIB}/libglog.so)
#set(GLOG_LIBRARIES ${ENVS_LIB})
#set(GLOG_LIBRARY_DIRS ${ENVS_LIB})

#HDF5
set(HDF5_INCLUDE_DIRS   ${ENVS_INCLUDE})
set(HDF5_LIBRARIES  ${ENVS_LIB}/libhdf5.so  ${ENVS_LIB}/libhdf5_cpp.so ${ENVS_LIB}/libhdf5_fortran.so)
set(HDF5_HL_LIBRARIES  ${ENVS_LIB}/libhdf5_hl.so)

#glags
set(GFLAGS_INCLUDE_DIR   ${ENVS_INCLUDE})
set(GFLAGS_LIBRARY ${ENVS_LIB}/libgflags.so )


#atlas
set(Atlas_CLAPACK_INCLUDE_DIR  ${ENVS_INCLUDE})
set(Atlas_CBLAS_INCLUDE_DIR  ${ENVS_INCLUDE})
set(Atlas_CBLAS_LIBRARY ${ENVS_LIB}/libcblas.so)
set(Atlas_BLAS_LIBRARY ${ENVS_LIB}/libblas.so)
set(Atlas_LAPACK_LIBRARY ${ENVS_LIB}/liblapack.so)

#levelDB
set(LevelDB_INCLUDE  ${ENVS_INCLUDE})
set(LevelDB_LIBRARY ${ENVS_LIB}/libleveldb.so)


#lmdb
set(LMDB_INCLUDE_DIR  ${ENVS_INCLUDE})
set(LMDB_LIBRARIES ${ENVS_LIB}/liblmdb.so)

#protobuf
set(PROTOBUF_INCLUDE_DIR  ${ENVS_INCLUDE})
#set(PROTOBUF_LITE_LIBRARIES ${ENVS_LIB}/libprotobuf-lite.so)
set(Protobuf_LIBRARY  ${ENVS_LIB}/libprotobuf-lite.so ${ENVS_LIB}/libprotoc.so ${ENVS_LIB}/libprotobuf.so)
#set(PROTOBUF_PROTOC_LIBRARIES ${ENVS_LIB}/libprotoc.so)
set(PROTOBUF_PROTOC_EXECUTABLE ${ENVS_EXECUTABLE}/protoc)

#boost
set(Boost_INCLUDE_DIR   ${ENVS_INCLUDE})
set(Boost_LIBRARIES  ${ENVS_LIB})

#MKL
set(MKL_INCLUDE_DIR  ${ENVS_INCLUDE})
set(MKL_LIBRARIES ${ENVS_LIB}/libmkl_avx2.so
        ${ENVS_LIB}/libmkl_avx512.so
        ${ENVS_LIB}/libmkl_avx.so
        ${ENVS_LIB}/libmkl_core.so
        ${ENVS_LIB}/libmkl_def.so
        ${ENVS_LIB}/libmkl_intel_ilp64.so
        ${ENVS_LIB}/libmkl_intel_lp64.so
        ${ENVS_LIB}/libmkl_mc3.so
        ${ENVS_LIB}/libmkl_mc.so
        ${ENVS_LIB}/libmkl_rt.so
        ${ENVS_LIB}/libmkl_sequential.so
        ${ENVS_LIB}/libmkl_vml_avx2.so
        ${ENVS_LIB}/libmkl_vml_avx.so
        ${ENVS_LIB}/libmkl_vml_cmpt.so
        ${ENVS_LIB}/libmkl_vml_def.so
        ${ENVS_LIB}/libmkl_vml_mc2.so
        ${ENVS_LIB}/libmkl_vml_mc3.so
        ${ENVS_LIB}/libmkl_vml_mc.so)

#openblas
SET(OpenBLAS_INCLUDE_DIR  ${ENVS_INCLUDE})
SET(OpenBLAS_LIB ${ENVS_LIB})

#snappy
set(Snappy_INCLUDE_DIR  ${ENVS_INCLUDE})
set(Snappy_LIBRARIES ${ENVS_LIB}/libsnappy.so)


# message("-- Warning: LIBCXX = ${LIBCXX}")
# if(LIBCXX STREQUAL "libstdc++11")
#     add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
# 如果最后link报protocbuf错误请添加该指令
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# endif()
#####################################################################################################################

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

#检查libcaffe.so 的link路径是否设置到miniconda的envs中
message("Caffe_INCLUDES: ${Caffe_INCLUDE_DIRS}")
message("Caffe_LINKER_LIBS: ${Caffe_LINKER_LIBS}")
```

# 3. 解析CMakeList
```c
cmake_minimum_required(VERSION 2.8.7)
```

设定cmake最低版本。高版本cmake提供更多的功能（例如cmake3.13开始提供target_link_directories()）或解决bug（例如OpenMP的设定问题），低版本有更好的兼容性。VERSION必须大写，否则不识别而报错。非必须但常规都写。放在最开始一行。

---

```c
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
```
cmake中也有if判断语句，需要配对的endif()。
POLICY是策略的意思，cmake中的poilcy用来在新版本的cmake中开启、关闭老版本中逐渐被放弃的功能特性：

---

```c
project(Caffe C CXX)
```
project()指令，给工程起名字，这列还写明了是C/C++工程，其实没必要写出来，因为CMake默认是开启了这两个的。这句命令执行后，自动产生了5个变量：
* ```PROJECT_NAME```，值等于```Caffe```
* ```PROJECT_SOURCE_DIR```，是```CMakeLists.txt```所在目录，通常是项目根目录（奇葩的项目比如protobuf，把CMakeLists.txt放在cmake子目录的也有）
* ```PROJECT_BINARY_DIR```，是执行cmake命令时所在的目录，通常是build一类的用户自行创建的目录。
* ```Caffe_SOURCE_DIR``` ，此时同 ```PROJECT_SOURCE_DIR```
* ```Caffe_BINARY_DIR``` ，此时同 ```PROJECT_BINARY_DIR```

---

```c
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
```

```set()```指令是设定变量的名字和取值， ```CACHE``` 意思是缓存类型，是说在外部执行CMake时可以临时指定这一变量的新取值来覆盖cmake脚本中它的取值：```CMAKE -Dvar_name=var_value```

而最后面的双引号包起来的取值可以认为是”注释“。STRING是类型，不过据我目前看到和了解到的，CMake的变量99.9%是字符串类型，而且这个字符串类型变量和字符串数组类型毫无区分。

变量在定义的时候直接写名字，使用它的时候则需要用```${VAR_NAME}```的形式。此外还可以使用系统的环境变量，形式为```$ENV{ENV_VAR_NAME}```，例如```$ENV{PATH}```，```$ENV{HOME}```等。

除了缓存变量，```option()```指令设定的东西也可以被用CMake -Dxxx=ON的形式来覆盖。

---

```add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})```
```add_definitions()```命令通常用来添加C/C++中的宏，例如：

```add_defitions(-DCPY_ONLY)``` ，给编译器传递了预定义的宏```CPU_ONLY```，相当于代码中增加了一句```#define CPU_ONLY```

```add_defitions(-DMAX_PATH_LEN=256)```，则相当于```#define MAX_PATH_LEN 256```
根据文档，实际上a```dd_definitions()```可以添加任意的编译器flags，只不过像添加头文件搜索路径等flags被交给```include_directory()```等命令了。

在这里具体的作用是，设定```CAFFE_VERSION```这一C/C++宏的值为```CAFFE_TARGET_VERSION```变量的取值，而这一变量在前面分析过，它是缓存变量，有一个预设的默认值，也可以通过```cmake .. -DCAFFE_TARGET_VERSION=x.y.z```来指定为```x.y.z```。

---
```list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)```

这里首先是```list(APPEND VAR_NAME VAR_VALUE)```这一用法，表示给变量```VAR_NAME```追加一个元素```VAR_VALUE```。虽然我写成```VAR_NAME```，但前面有提到，cmake中的变量几乎都是字符串或字符串数组，这里VAR_NAME你就当它是一个数组就好了，而当后续使用```${VAR_NAME}```时输出的是”整个数组的值“。（吐槽：这不就是字符串么？为什么用list这个名字呢？搞得像是在写不纯正的LIPS）

具体的说，这里是把项目根目录(CMakeLists.txt在项目根目录，```${PROJECT_SOURCE_DIR}```表示CMakeLists.txt所在目录）下的cmake/Modules子目录对应的路径值，追加到```CMAKE_MODULE_PATH```中；```CMAKE_MODULE_PATH```后续可能被```include()```和```find_package()```等命令所使用。

---

```c
include(ExternalProject)
include(GNUInstallDirs)
```

`include()`命令的作用：
* 包含文件，
* 或者，包含模块
* 所谓**包含文件**，例如```include(utils.cmake)```，把当前路径下的```utils.cmake```包含进来，基本等同于C/C++中的```#include```指令。通常，include文件的话文件应该是带有后缀名的。
* 所谓**包含模块**，比如```include(xxx)```，是说在```CMAKE_MODULE_PATH```变量对应的目录，或者CMake安装包自带的Modules目录（比如mac下brew装的cmake对应的是```/usr/local/share/cmake/Modules```)里面寻找```xxx.cmake```文件。注意，此时不需要写".cmake"这一后缀。

具体的说，这里是把CMake安装包提供的`ExternalProject.cmake`(例如我的是```/usr/local/share/cmake/Modules/ExternalProject.cmake```）文件包含进来。ExternalProject，顾名思义，引入外部工程，各种第三方库什么的都可以考虑用它来弄;

GNUInstallDirs也是对应到CMake安装包提供的GNUInstallDirs.cmake文件，这个包具体细节还不太了解，可自行翻阅该文件。

---

```c
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)
```

这里是实打实的包含了在项目cmake子目录下的5各cmake脚本文件了，是Caffe作者们（注意，完整的Caffe不是Yangqing Jia一个人写的）提供的，粗略看了下：

* ```cmake/Utils.cmake: ```定义了一些通用的（适用于其他项目的）函数和宏，用于变量（数组）的打印、合并、去重、比较等
* ```cmake/Targets.cmake:``` 定义了Caffe项目本身的一些函数和宏，例如源码文件组织、目录组织等。
* ```cmake/Misc.cmake```：杂项，比较抠细节的一些设定，比如通常CMAKE_BUILD_TYPE基本够用了，但是这里通过CMAKE_CONFIGURATION_TYPES来辅助设定CMAKE_BUILD_TYPE，等等
* ``` cmake/Summary.cmake```：定义了4个打印函数，用来打印Caffe的一些信息，执行CMake时会在终端输出，相比于散落在各个地方的message()语句会更加系统一些
* `cmake/ConfigGen.cmake`: 整个caffe编译好之后，如果别的项目要用它，那它也应该用cmake脚本提供配置信息。

---
```c
# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" ON) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)
```
这里是设定各种`option`，也就是”开关“，然后后续根据开关的取值（布尔类型的变量，利用if和else来判断），编写各自的构建规则。
其中`caffe_option()`是`cmake/Utils.cmake`中定义的，它相比于cmake自带的`option()`命令，增加了可选的条件控制字段：

```c
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   caffe_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
```

---

```c
# ---[ Dependencies
include(cmake/Dependencies.cmake)
```
这里是包含`Dependencies.cmake`，它里面配置了Caffe的绝大多数依赖库：
>Boost
Threads
OpenMP
Google-glog
Google-gflags
Google-protobuf
HDF5
LMDB
LevelDB
Snappy
CUDA
OpenCV
BLAS
Python
Matlab
Doxygen

其中每一个依赖库库都直接（在Dependencies.cmake中）或间接（在各自的cmake脚本文件中）使用`find_package()`命令来查找包。

1. `find_package(Xxx)`如果执行成功，则提供相应的`Xxx_INCLUDE_DIR`、`Xxx_LIBRARY_DIR`等变量，看起来挺方便，但其实并不是所有的库都提供了同样的变量后缀，其实都是由库的官方作者或第三方提供的xxx.
2. `find_packge(Xxx)`实际中往往是翻车重灾区。它其实有N大查找顺序，而CSDN上的博客中往往就瞎弄一个，你照搬后还是不行。具体例子：

---
```c
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()
```
通过设定`CMAKE_CXX_FLAGS`，cmake生成各自平台的makefile、.sln或xcodeproject文件时设定同样的CXXFLAGS给编译器。如果是.c文件，则由c编译器编译，对应的是`CMAKE_C_FLAGS`。

这里的set()指令设定`CMAKE_CXX_FLAGS`的值，加入了两个新的flags："-fPIC"和"-Wall"。实际上用`list(APPEND CMAKE_CXX_FLAGS "-fPIC -Wall")`是完全可以的。set()只不过是有时候可能考虑设定变量默认值的时候用一用。

`-fPIC`作用于编译阶段，告诉编译器产生与位置无关代码(Position-Independent Code)，则产生的代码中，没有绝对地址，全部使用相对地址，故而代码可以被加载器加载到内存的任意位置，都可以正确的执行。这正是共享库所要求的，共享库被加载时，在内存的位置不是固定的。

-Wall则是开启所有警告。根据个人的开发经验，C编译器的警告不能完全忽视，有些wanring其实应当当做error来对待，例如
- 函数未定义而被使用（忘记#include头文件）
- 指针类型不兼容(incompatible)
- 都有可能引发seg fault，甚至bus error。

---

`caffe_set_caffe_link()`

这里是设置Caffe_LINK这一变量，后续链接阶段会用到。它定义在cmake/Targets.cmake中：

```c
# Defines global Caffe_LINK flag, This flag is required to prevent linker from excluding
# some objects which are not addressed directly but are registered via static constructors
macro(caffe_set_caffe_link)
  if(BUILD_SHARED_LIBS)
    set(Caffe_LINK caffe)
  else()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(Caffe_LINK -Wl,-force_load caffe)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(Caffe_LINK -Wl,--whole-archive caffe -Wl,--no-whole-archive)
    endif()
  endif()
endmacro()
```
可以看到，如果是编共享库（动态库），则就叫caffe；否则，则增加一些链接器的flags：-Wl是告诉编译器，后面紧跟的是链接器的flags而不是编译器的flags（现在的编译器往往是包含了调用连接器的步骤）。

---
```c
if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()
```
在前面已经`include(cmake/Dependencies.cmake)`的情况下，`Dependencies.cmake`中的`include(cmake/Cuda.cmake)`使得Cuda的设定也被载入。而`Cuda.cmake`中的最后，判断如果当前操作系统是苹果系统并且>10.8、cuda版本小于7.0，那么使用`libstdc++`而不是`libc++`：
```c
# Handle clang/libc++ issue
if(APPLE)
  caffe_detect_darwin_version(OSX_VERSION)

  # OSX 10.9 and higher uses clang/libc++ by default which is incompatible with old CUDA toolkits
  if(OSX_VERSION VERSION_GREATER 10.8)
    # enabled by default if and only if CUDA version is less than 7.0
    caffe_option(USE_libstdcpp "Use libstdc++ instead of libc++" (CUDA_VERSION VERSION_LESS 7.0))
  endif()
endif()]
```
---
```c
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)
```
这里添加的编译器flags，是用来屏蔽特定类型的警告的。虽说眼不见心不烦，关掉后少些warning输出，但是0error0warning不应该是中级目标吗？

---
```c
# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

```
这是设定configure file。configure_file()命令是把输入文件（第一个参数）里面的一些内容做替换（比如${var}和@var@替换为具体的值，宏定义等），然后放到指定的输出文件（第二个参数）。其实还有其他没有列出的参数。

具体说，这里生成了build/caffe_config.h，里面define了几个变量：
```c
/* Sources directory */
#define SOURCE_FOLDER "${PROJECT_SOURCE_DIR}"

/* Binaries directory */
#define BINARY_FOLDER "${PROJECT_BINARY_DIR}"

/* This is an absolute path so that we can run test from any build
 * directory */
#define ABS_TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/src/caffe/test/test_data/"

/* Test device */
#define CUDA_TEST_DEVICE ${CUDA_TEST_DEVICE}
```
---

```c
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})
```
这里是设定两个自定义变量Caffe_INCLUDE_DIR和Caffe_SRC_DIR的值，只不过它俩比较特殊，想想：
如果以后别人find_package(Caffe)，其实就需要其中的Caffe_INCLUDE_DIR的值。anyway，那些是后续export命令干的事情，这里忽略。

这里第三句include_directories()命令，把build目录加入到头文件搜索路径了，其实就是为了确保caffe_config.h能被正常include（就一个地方用到它）：

---
```c
# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected    definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR}                         ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()
```
擦亮眼睛：Caffe的cmake脚本中分别定义了`Caffe_INCLUDE_DIR`和`Caffe_INCLUDE_DIRS`两个变量，只相差一个S，稍不留神容易混掉：不带S的值是`$Caffe_ROOT/include`，带S的值是各个依赖库的头文件搜索路径（在Dependencies.cmake中多次list(APPEND得到的。类似的，`Caffe_DEFINITIONS`也是在Dependencies.cmake中设定的。

这里判断出如果有CUDA的话就把`Caffe_INCLUDE_DIRS`变量中的`PUBLIC`和`PRIVATE`都去掉，把`Caffe_DEFINITIONS`中的PUBLIC和PRIVATE也去掉。

---

```c
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)
```
使用`add_subdirectory()`，意思是说把子目录中的`CMakeLists.txt`文件加载过来执行，从这个角度看似乎等同于`include()`命令。实则不然，因为它除了按给定目录名字后需要追加"/CMakeLists.txt"来构成完整路径外，往往都是包含一个target(类似于git中的submodule了），同时还可以设定别的一些参数：

- 指定binary_dir
- 设定EXCLUDE_FROM_ALL，也就是”搞一个独立的子工程“，此时需要有project()指令，并且不被包含在生成的.sln工程的ALL目标中，需要单独构建。
  
粗略看看各个子目录都是做什么的：
- `src/gtest`，googletest的源码
- `src/caffe`，caffe的源码构建，因为前面做了很多操作（依赖库、路径，etc），这里写的就比较少。任务只有2个：构建一个叫做caffe的库，以及test。
- `tools`，这一子目录下每一个cpp文件都生成一个xxx.bin的目标，而最常用的就是caffe训练接口build/caffe这个可执行文件了。
- `examples`，这一子目录下有cpp_classification的C++代码，以及mnist，cifar10，siamse这三个例子的数据转换的代码，这四个都是C++文件，每一个都被编译出一个可执行
- `python`，pycaffe接口，python/caffe/_caffe.cpp编译出动态库
- `matlab`，matlab接口，./+caffe/private/caffe_.cpp编译出？编译出一个定制的目标，至于是啥类型，也许是动态库吧，玩过matlab和C/C++混编的都知道，用mex编译C/- C++为.mexa文件，然后matlab调用.mexa文件，其实就是动态库
- `docs`，文档，doxygen、jekyll都来了，以及拷贝每一个.ipynb文件。没错，add_custom_command()能定制各种target，只要你把想要执行的shell脚本命令用cmake的语法来写就可以了，很强大。

---
```c
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

```
`add_custom_command()`能定制各种target，只要你把想要执行的shell脚本命令用cmake的语法来写就可以了，很强大。

这里依然是定制的target，具体看来是调用scripts/cpplint.py(谷歌官方C++代码风格检查工具）来执行代码风格检查。（个人觉得G家的C++风格有一点不太好：缩进两个空格太少了，费眼睛，强烈建议和Visual Studio保持一致，用tab并且tab宽度为4个空格）。

所谓linter就是语法检查器，除了cpplint其实还可以用cpp_check、gcc、clang等，我的vim中配置的就是用cpp_check和gcc，

--- 
```c
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()
```
如果开启了`BUILD_python`开关，那么执行一个定制的target（执行pytest）。
`add_dependencies()`意思是指定依赖关系，这里要求`pycaffe`目标完成后再执行`pytest`目标，因为pytest需要用到`pycaffe`生成的caffe模块。`pycaffe`在前面提到的`add_subdirectory(python)`中被构建。

--- 
```c
# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()
```

在Caffe根目录的CMakeLists.txt的最后，是打印各种配置的总的情况，以及输出各种配置(后者其实包含了install()指令的调用)

---
```c
# Dependencies.cmake

# ---[ Python
if(BUILD_python)

  message(STATUS "var: " ${python_version})
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    message(status "wf log1 !!")
    # use python3
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
    find_package(NumPy 1.7.1)
    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})

    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
	# string 的REGEX REPLACE尽可能地多次匹配正则表达式，并用# 
	#replace_expression取代输出中的匹配项。在匹配之前连接所有<input>参数。替
	# 换表达式可以使用\1, \2, …, \9引用匹配的括号分隔的子表达式。



    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
      STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

      STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
      find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
      set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

      STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
      if("${has_more_version}" STREQUAL "")
        break()
      endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
      find_package(Boost 1.46 COMPONENTS python)
    endif()
  else()
    # disable Python 3 search
    find_package(PythonInterp 2.7)
    find_package(PythonLibs 2.7)
    find_package(NumPy 1.7.1)
    find_package(Boost 1.46 COMPONENTS python)
  endif()
  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    if(BUILD_python_layer)
      list(APPEND Caffe_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)
      list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} PUBLIC ${Boost_INCLUDE_DIRS})
      list(APPEND Caffe_LINKER_LIBS PRIVATE ${PYTHON_LIBRARIES} PUBLIC ${Boost_LIBRARIES})
    endif()
  endif()
endif()
```


# 4. 











