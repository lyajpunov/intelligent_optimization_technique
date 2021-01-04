
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
 
def fit_fun(X):  # 适应函数
    A = 10
    return 2 * A + X[0] ** 2 - A * np.cos(2 * np.pi * X[0]) + X[1] ** 2 - A * np.cos(2 * np.pi * X[1])
 
class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子在i维空间的位置
        # 粒子的位置，随机赋予在区间（-x_max, x_max）初值
        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子在i维空间的速度
        # 粒子的速度，随机赋予在区间（-x_max, x_max）初值
        self.__bestPos = [0.0 for i in range(dim)]  
        # 粒子在i个维度上最好的位置，将每个方向的粒子都赋予0.0
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值
 
    def set_pos(self, i, value):
        self.__pos[i] = value #位置/解
 
    def get_pos(self):    #返回位置/解
        return self.__pos
 
    def set_best_pos(self, i, value):#粒子最优位置/解
        self.__bestPos[i] = value  
 
    def get_best_pos(self):
        return self.__bestPos#返回粒子位置/解
 
    def set_vel(self, i, value):#速度
        self.__vel[i] = value 
 
    def get_vel(self):
        return self.__vel#返回速度
 
    def set_fitness_value(self, value):
        self.__fitnessValue = value   #粒子适应度
 
    def get_fitness_value(self):
        return self.__fitnessValue   #返回粒子适应度
 
 
class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, C1, C2, W , best_fitness_value=float('Inf'), Gamma=1.4):
        self.C1 = C1
        self.C2 = C2
        self.Gamma= Gamma #调整迭代的速度
        self.W = W        #惯性权重
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max     #最大解/位置
        self.max_vel = max_vel  # 粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
 
        # 对粒子群所有粒子参数进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]
 
    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value
 
    def get_bestFitnessValue(self):
        return self.best_fitness_value
    # 返回粒子群最优适应度，即本代最优粒子适应度
 
 
    def set_bestPosition(self, i, value):
        self.best_position[i] = value
      # 更新本代粒子群最优位置，令本代最优粒子的位置/解为本代粒子群最优位置
 
 
    def get_bestPosition(self):
        return self.best_position
    # 返回粒子群最优位置，即本代最优粒子的位置/解
 
 
    # 更新粒子速度
    def update_vel(self, part):
        for i in range(self.dim):#更新粒子第i维的速度
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            #权重*上代粒子第i维度的速度+c1*random*(粒子上代在i维度的最优位置-粒子当前在第i维度的最位置)
            #   +c2*random*(上代粒子群在i维的最优位置-粒子当前在第i维度的最位置)
            if vel_value > self.max_vel:
                vel_value = self.max_vel   #限速
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)  #在各个维度更新位置
 
    # 更新粒子的位置
    def update_pos(self, part):
        for i in range(self.dim):   #更新粒子第i维度的位置
            pos_value = part.get_pos()[i] + self.Gamma*part.get_vel()[i]
            part.set_pos(i, pos_value)#位置更新公式
        value = fit_fun(part.get_pos())#更新粒子解的适应度
        if value < part.get_fitness_value():#如果粒子的适应度小于上代粒子适应度
            part.set_fitness_value(value) #则保留
            for i in range(self.dim):    
                part.set_best_pos(i, part.get_pos()[i])
                #将i维中的最优位置作为粒子的最优位置,如果粒子的适应度大于上代粒子适应度则抛弃
 
 
        if value < self.get_bestFitnessValue():
            #如果粒子的适应度优于于粒子群全局适应度，则保留
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])
                #更新粒子群最优位置
         
    def update(self):    #更新粒子群
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue()) 
        # 每次迭代完把当前的最优适应度存到列表/即，把最优函数极值保存
        return self.fitness_val_list, self.get_bestPosition()


def  main():
    # 维度
    dim = 2
    # 种群规模
    size = 50
    # 迭代次数
    iter_num = 150
    # 范围
    x_max = 5.12
    # 最大速度
    max_vel = 0.5
    # 惯性权重
    w = 1
    # 控制参数
    c1 = 2
    c2 = 2
     
    pso = PSO(dim, size, iter_num, x_max, max_vel,c1,c2,w)  #初始化
    fit_var_list, best_pos = pso.update()   #执行PSO
    print("迭代次数：" + str(iter_num))
    print("最优解:" + str(best_pos))  
    print("最优值:" + str(fit_var_list[-1])) 
    plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list, c="b", alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()