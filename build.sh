cd build

cmake -DCMAKE_C_COMPILER=/home/koi/install/loongson-gnu-toolchain/bin/loongarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/home/koi/install/loongson-gnu-toolchain/bin/loongarch64-linux-gnu-g++  .. && make -j16

cd ..
