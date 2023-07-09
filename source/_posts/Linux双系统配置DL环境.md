---
title: Linux双系统配置DL环境
date: 2023-07-09 14:36:50
tags: 
    - Linux
    - Cuda
    - 代理

categories: 
    - 操作指南

mathjax: true
---
本文记录了在同一台电脑上安装Windows和Linux双系统并配置深度学习环境过程中一些关键要点。

# 1. 安装Linux双系统

安装过程参考此篇博文[装了5次Ubuntu，告诉你win10+Ubuntu双系统的正确打开方式 - czpcalm的文章 - 知乎](https://zhuanlan.zhihu.com/p/101307629)。 

需要注意的是，建议boot分区和/分区划分大一些。

# 2. 安装Cuda和cuDNN

## 2.1 安装Nvidia驱动
安装过程参考此篇博文[Linux 下的 CUDA 安装和使用指南 - 知乎用户XXdFO7的文章 - 知乎](https://zhuanlan.zhihu.com/p/79059379)。

如果安装驱动之后输入nvidia-smi出现 "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
可以尝试参考下面博文
[Solved NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.](https://clay-atlas.com/us/blog/2022/07/29/solved-nvidia-smi-has-failed-because-it-couldnt-communicate-with-the-nvidia-driver-make-sure-that-the-latest-nvidia-driver-is-installed-and-running/)。其核心思想就是先移除已安装的驱动，再安装合适的版本驱动。

## 2.2 安装Cuda和cuDNN

这里主要陈述时安装多个版本的cuda情况。可以参考如下博客[ubuntu多个cuda与cudnn版本切换](https://blog.csdn.net/weixin_42070745/article/details/113621393)

核心要义是环境变量配置为/usr/local/cuda，当切换时直接删除/usr/local/cuda，然后创建不同版本cuda指向cuda的软连接。

# 3 Yooyun666 linux下的代理

严格按照网站给的教程来走即可，注意，所谓的clash-dashboard是一个网址，我们点击clash-board超链接即可。
