---
title: CLAPACK学习
date: 2017-12-23
toc: true
categories: 工具学习
tags: [lapack, clapack, 线性代数工具包]
description: 一个开源的线性代数工具包，可用于求解线性方程组、线性最小二乘、特征值和奇异值等相关问题
mathjax: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX"],
    tex2jax: {
      inlineMath: [ ['$','$'], ['\\(','\\)'] ],
      displayMath: [ ['$$','$$']],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML,http://myserver.com/MathJax/config/local/local.js">
</script>


**LAPACK**：全称Linear Algebra PACKage，美国国家科学基金等资助开发的著名公开软。以Fortran语言编写，提供了丰富函数，用于求解线性方程组、线性最小二乘、特征值和奇异值等相关问题。spark mllib, mxnet等都在底层使用了lapack进行相关的线性代数计算。

**CLAPACK**：使用f2c工具将LAPACK的Fortran代码转换成C语言代码的C语言算法包, 可用于在C语言环境中直接调用线性代数的相关函数功能。

本文主要是描述如何在linux环境中安装clapack，以CLAPACK-3.2.1为例进行说明，并使用clapack实现cholesky分解过程。

# clapack安装
## 准备安装文件
* 远程获取文件：wget http://www.netlib.org/clapack/clapack.tgz  
* 将clapack.tgz拷贝到准备安装的目录，运行tar -xvf clapack.tgz 完成解压。
* cd CLAPACK-3.2.1 进入CLAPACK主目录。
* cp make.inc.example make.inc 

**此时目录下的主要文件目录：**

* BLAS：blas C语言源码，clapack需要调用的该底层函数库。
* F2CLIBS：f2c相关函数库
* INCLUDE：clapack, blas, f2c库对应的头文件
* INSTALL：测试函数，对于不同的平台提前测试make.inc对应的配置
* Makefile：构建文件
* make.inc：定义compiler, compile flags and library。
* SRC：LAPACK c语言代码，当我们要查某个函数的具体参数时，可以到这个目录下根据函数的名字找到对应的.c文件
* TESTING：用于对clapack函数测试其正确性

## 安装
* 编译f2c: make f2clib
* 编译blas: make blaslib, 需要注意的是，这里是使用的该clapck包所引用的blas库，没有针对所有机器做优化。如果想针对自己的机器，使用对应的库使速度达到最优，可以参考BLAS/WRAP目录
* 运行blas测试程序：
  cd BLAS/TESTING && make -f Makeblat2
  cd ..
  ./xblat2s < sblat2.in
  ./xblat2d < dblat2.in
  ./xblat2c < cblat2.in
  ./xblat2z < zblat2.in
  cd TESTING && make -f Makeblat3
  cd ..
	./xblat3s < sblat3.in
	./xblat3d < dblat3.in
	./xblat3c < cblat3.in
	./xblat3z < zblat3.in
  cd ..
* 修改make.inc:
  CC        = gcc
  BLASLIB      = ../../blas$(PLAT).a

* 编译clapack源码及相关测试：
  cd INSTALL && make && cd ..
  cd SRC/ && make && cd ..
  cd TESTING/MATGEN && make && cd .. && make
  上述步骤都通过后，在主目录下生成blas_LINUX.a， lapack_LINUX.a二个库，其他程序调用时通过引用这两个库，调用clapack完成线性代数相关计算。  
 
# 测试
   为测试环境可正常使用，此处使用clapack，利用cholesky分解求解线性方程组。
   方程组如下：
   ```
   A = “4.16   -3.12  0.56  -0.10
        -3.12  5.03   -0.83 1.18
        0.56   -0.83  0.76  0.34
        -0.10  1.18   0.34  1.18”
                          
   b = "8.7
        -13.35
        1.89
        -4.14"
   求解 Ax = b方程组
   ```
   
   c++代码：
   ```c++
#include <iostream>
#include <fstream>
#include "blaswrap.h"
#include "f2c.h"
#include "clapack.h"

using namespace std;

int main(int argc, char** argv){

  long int k = 4;
  long int nrhs = 1;
  long int ldb = k;
  long int info = 0;
  double a[10]={4.16, -3.12, 5.03, 0.56, -0.83, 0.76, -0.10, 1.18, 0.34, 1.18};
  double b[4]={8.7, -13.35, 1.89, -4.14};
  
  char matrix_type='U'; 
  dppsv_(&matrix_type, &k, &nrhs, a, b, &k, &info);
  cout << "solution:";
  for (int i=0; i< k; i++){
      cout << b[i] << " ";
  }
  cout << endl;
  return 0;
}
   ```
**编译程序：**
在目录下新建文件夹example, mkdir example
保存文件main.cc
g++ -o main main.cc -I ../INCLUDE -L ../ -lblas -llapack   
(-I和-L选项需要根据自己机器对应的头文件和库文件目录来写)

**运行程序：**
./main

输入如下： solution:1 -1 2 -3
 
# 参考资料
【1】 lapack@cs.utk.edu, http://www.netlib.org/clapack/


