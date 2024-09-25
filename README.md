本文旨在记录学习吴恩达老师的机器学习实验的过程（课程在b站：https://www.bilibili.com/video/BV1Bq421A74G?p=1&vd_source=43478a57ace1753f8a10a9b342b5e5b6）

## 1.安装与使用jupyter

### 安装pip

一般来讲，安装`python`的时候就会直接安装`pip`，我们需要用`pip`来安装`jupyter`

检查有没有`pip`，打开终端输入命令：

```shell
pip --version
```

如果出现`pip`的版本号，则说明安装成功

![QQ_1727244354465](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727244354465.png)

### 安装jupyter

```shell
pip3 install jupyter
```

同样的安装以后，可以通过查看版本号检查是否安装成功：

```shell
jupyter --version
```

![QQ_1727244647680](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727244647680.png)

### 使用终端启动jupyter

在终端中输入`jupyter notebook`并按回车执行。执行命令后，终端会输出一系列信息，包括Jupyter Notebook的启动地址（通常是`http://localhost:8888`）。同时，默认情况下，你的默认网页浏览器会自动打开这个地址，显示Jupyter Notebook的主界面

```shell
jupyter notebook
```

![QQ_1727244726277](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727244726277.png)

下面就是我们打开jupyter以后看到的页面

![QQ_1727244755265](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727244755265.png)

### 使用jupyter打开实验

在代码目录下有三大实验，每个实验有3-4周，以打开第一周的实验为例，找到work目录。

![QQ_1727245924929](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727245924929.png)

全选，然后右键，然后点击open

![QQ_1727245941358](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727245941358.png)

下面，我们就用jupyter打开了实验

![QQ_1727245954124](/Users/liujingmin/Library/Containers/com.tencent.qq/Data/tmp/QQ_1727245954124.png)