---
title: Hexo博客迁移
date: 2023-06-08 21:59:31
tags: 
    - Hexo

categories: 
    - 操作指南

mathjax: true
---

# 1. Git仓库介绍

Git仓库包含两个分支，main和src_code，其中src_code存储的是markdown源代码，main分支存储的是markdown代码生成的网页源码。

# 2. 环境准备

需要安装NodeJs，Pandoc

# 3. 操作指令

+ 新建一个文件夹 `GitBlog` 来保存博客源代码
+ 进入`GitBlog`，安装hexo：`npm install -g hexo-cli`
+ 将src_code克隆至 `GitBlog`
+ 安装hexo依赖: `npm install`。 如果有错误，如hexo-renderer-sass-next安装不上，可以单独安装 `npm install hexo-renderer-sass-next`
+ 如果有需要，可以安装额外依赖，如 `npm install hexo-deployer-git --save`

注意，如果 `hexo-renderer-pandoc`报错，需要安装Pandoc。Pandoc是渲染数学公式Mathjax必要的，不可移除。