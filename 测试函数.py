import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_pic(X, Y, Z, z_max, title, z_min=0):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    # ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    plt.savefig(r"E:\课程文件\网课\人工智能\论文\Rastrigin.png") # 保存图片
    plt.show()


def get_X_AND_Y(X_min, X_max, Y_min, Y_max):
    X = np.arange(X_min, X_max, 0.1)
    Y = np.arange(Y_min, Y_max, 0.1)
    X, Y = np.meshgrid(X, Y)
    return X, Y


#  rastrigin测试函数
def Rastrigin(X_min = -5.12, X_max = 5.12, Y_min = -5.12, Y_max = 5.12):
    A = 10
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    return X, Y, Z, 100, "Rastrigin function"


# Ackley测试函数
def Ackley(X_min = -5, X_max = 5, Y_min = -5, Y_max = 5):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20
    return X, Y, Z, 15, "Ackley function"


# Sphere测试函数
def Sphere(X_min = -10, X_max = 10, Y_min = -10, Y_max = 10):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = X**2 + Y**2
    return X, Y, Z, 100, "Sphere function"


#  beale测试函数
def Beale(X_min = -4.5, X_max = 4.5, Y_min = -4.5, Y_max = 4.5):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = np.power(1.5 - X + X * Y, 2) + np.power(2.25 - X + X * (Y ** 2), 2) \
        + np.power(2.625 - X + X * (Y ** 3), 2)
    return X, Y, Z, 150000, "Beale function"


# Booth测试函数
def Booth(X_min = -10, X_max = 10, Y_min = -10, Y_max = 10):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = np.power(X + 2*Y - 7, 2) + np.power(2 * X + Y - 5, 2)
    return X, Y, Z, 2500, "Booth function"


# Bukin测试函数
def Bukin(X_min = -15, X_max = -5, Y_min = -3, Y_max = 3):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = 100 * np.sqrt(np.abs(Y - 0.01 * X**2)) + 0.01 * np.abs(X + 10)
    return X, Y, Z, 200, "Bukin function"


#  Three-hump camel测试函数
def three_humpCamel(X_min = -5, X_max = 5, Y_min = -5, Y_max = 5):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = 2 * X**2 - 1.05 * X**4 + (1/6) * X**6 + X*Y + Y*2
    return X, Y, Z, 2000, "three-hump camel function"


# Hölder table测试函数
def Holder_table(X_min = -10, X_max = 10, Y_min = -10, Y_max = 10):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = -np.abs(np.sin(X) * np.cos(Y) * np.exp(np.abs(1 - np.sqrt(X**2 + Y**2)/np.pi)))
    return X, Y, Z, 0, "Hölder table function", -20



z_min = None
# 这里以Rastrigin测试函数为例
X, Y, Z, z_max, title = Rastrigin()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title = Ackley()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title = Sphere()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title = Beale()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title = Booth()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title = Bukin()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title = three_humpCamel()
draw_pic(X, Y, Z, z_max, title, z_min)
X, Y, Z, z_max, title, z_min = Holder_table()
draw_pic(X, Y, Z, z_max, title, z_min)
