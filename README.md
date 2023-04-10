# 一个非常简单的深度学习框架小功能的实现
在编写之前没有完整构思到每一块所以最终呈现出来的代码并不美观。同时有一些不必要的代码冗余没有删除，后续如果有时间（机会很少）会改进一下。希望没有给您带来不必要的麻烦！！

如果发现代码有错误可以在github上面发送消息，如果我看到且有时间回复一定会回复。由于c++用的也比较少所以可能很多地方的写法可能会比较幼稚，所以如果在代码中有不正确的的地方给您带来困扰请给我反馈，当然也欢迎给出您宝贵的修改意见。

如果是因为在写法上犯了一些比较弱智的“冗余操作”但没有逻辑错误的话请您多多包涵，如很多地方比如数据流图没必要完全传递，一些算子节点在计算过程中由于写法的问题造成了复杂度上升的问题您看看就好。。。

ps：[BILIBILI视频链接](https://www.bilibili.com/video/BV1Q24y1g7oj/?spm_id_from=333.337.search-card.all.click&vd_source=f817c0e82770e849e62c360c6d27fc4c)
## 1.环境要求

* MSVC
* Python 3.8+
* numpy
* [pybind11](https://github.com/pybind/pybind11)
* [xtensor](https://github.com/xtensor-stack/xtensor-blas)
* [xtl](https://github.com/xtensor-stack/xtl)
* [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas)
* [xtensor-python](https://github.com/xtensor-stack/xtensor-python)
* cmake
## 2 使用方法
### 2.1 源代码编写
如果想改写源代码，可以首先将（pybind11,xtensor,xtl,xtensor-blas,xtensor-python）这几个库下载到lib文件夹中，然后去改写src文件中的内容，编译的话windows平台推荐msvc，创建build文件夹后执行文件夹最外层的cmakelist文件即可。
### 2.2 运行该框架的demo
Pypanda中是已经编译好的动态库以及python接口代码，在test.py中包含了b站视频中出现的3个demo，如果您用python3.9的虚拟环境且在有numpy和pybind11支持的情况下执行test.py应该是可以运行的，这里我只在自己的台式机和笔记本电脑上测试过，如果您的numpy版本不对也可能造成无法运行。

