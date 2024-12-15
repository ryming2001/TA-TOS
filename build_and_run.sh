#!/bin/bash

clear 

# 设置工作目录为项目根目录
cd "$(dirname "$0")"

# 创建或进入build目录
if [ ! -d "build" ]; then
  mkdir build
fi
cd build



# 运行cmake，指定构建类型为Release
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译项目
make -j$(nproc)

# 如果编译成功，运行可执行文件
if [ $? -eq 0 ]; then
  echo "Running the program..."
  ./devel/lib/TATOS_cpp_v1/tatos
else
  echo "Build failed!"
fi

