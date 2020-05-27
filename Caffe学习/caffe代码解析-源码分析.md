# 1. MNIST数据集

## 1.1 下载
> get_mnist.sh
```shell
#！/bin/bash
# #!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
```

> create_mnist.sh
```shell
#！/bin/bash
# #!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
# 执行的时候如果出现了返回值为非零，整个脚本 就会立即退出
set -e
# lmdb/leveldb 的生成路劲
EXAMPLE=examples/mnist
DATA=data/mnist
BUILD=build/examples/mnist

# 后端类型，可选 lmdb/leveldb
BACKEND="lmdb"

#
echo "Creating ${BACKEND}..."

# 如果已经存在，则先删除
rm -rf $EXAMPLE/mnist_train_${BACKEND}
rm -rf $EXAMPLE/mnist_test_${BACKEND}

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
```
 调用了`convert_mnist_data.bin`程序，查看源码：
```c++


```


















