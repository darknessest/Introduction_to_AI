import numpy as np
from numpy import sqrt, exp
import random
from time import perf_counter

scaleValue = 32767
InputNo = 3
OutputnNo = 3
PatternNo = 358
ErrorLevelValue = 0.005  # 满意的误差水平
wmaxValue = 0.1

HiddenNo = 10
IterationsNo = 50000
etaValue = 0.1  # 默认的学习速率
alphaValue = 0.1  # 默认的势态因子

nReportErrors = 100

MonitorError = False  # 表示为真

error1 = 0.0  # 最新的误差平方之和

# temp  # 临时变量

out1 = np.zeros([PatternNo, HiddenNo])              # 隐含层, [模式数][隐含]
delta1 = np.zeros([PatternNo, HiddenNo])            # 隐含层中的误差修正量, [模式数][隐含]
delw1 = np.zeros([HiddenNo + 1, InputNo + 1])       # 输入与隐含层之间权值的变化, [隐含 + 1][输入 + 1]
w1 = np.zeros([HiddenNo + 1, InputNo + 1])          # 输入与隐含层之间的权值, [隐含 + 1][输入 + 1]
out2 = np.zeros([PatternNo, OutputnNo])             # 输出层
delta2 = np.zeros([PatternNo, OutputnNo])           # 输出层中的误差修正量
delw2 = np.zeros([OutputnNo, HiddenNo + 1])         # 隐含与输出层之间权值的变化, [输出][隐含 + 1]
w2 = np.zeros([OutputnNo, HiddenNo + 1])            # 隐含与输出层间的权值, [输出][隐含 + 1]
x = np.zeros([PatternNo, InputNo])                  # 历史数据输入值
y = np.zeros([PatternNo, OutputnNo])                # 历史数据输出值
out0 = np.zeros([PatternNo + 2, InputNo])
target = np.zeros([PatternNo + 2, OutputnNo])
ss = np.zeros([InputNo])
average = np.zeros([InputNo])
sss = np.zeros([InputNo])
fangcha = np.zeros([InputNo])
ssy = np.zeros([InputNo])
averagey = np.zeros([InputNo])
sssy = np.zeros([InputNo])
fangchay = np.zeros([InputNo])
MIN = 0.0
MAX = 0.0

h = 0                       # 代表隐含层的值
i = 0                       # 代表输入层的值
j = 0                       # 代表输出层的值
p = 0                       # 模式数
qq = 0                      # 循环值

nPatterns = PatternNo       # 模式个数
nInputNodes = InputNo       # 输入节点个数
nHiddenNodes = HiddenNo     # 隐含层节点个数
nOutputNodes = OutputnNo    # 输出层节点个数

# 将节点数写到数据文件中去
my_file3 = open("节点数.dat", "w")

my_file3.write("")
my_file3.write(str(nInputNodes))
my_file3.write(str(nHiddenNodes))
my_file3.write(str(nOutputNodes))
my_file3.close()

nIterations = IterationsNo  # 迭代次数

frand = 0

scale = scaleValue
errorLimit = ErrorLevelValue  # 满意的误差水平
wmax = wmaxValue
eta = etaValue  # 默认的学习速率
alpha = alphaValue  # 默认的势态因子

# 一.神经网络的训练阶段

# 将历史数据从数据文件中读入数组中
# ifstream my_file1("S1期鼓风操作数据.dat")
my_file1 = open("S1期鼓风操作数据.dat", "r")
# my_file1.seekg(0, ios::beg)

for i in range(nPatterns):
    for j in range(nInputNodes + 2):
        if j >= nInputNodes:
            y[i][j - nInputNodes] = float(my_file1.readline())
        else:
            x[i][j] = float(my_file1.readline())

my_file1.close()

# 将输入与输出值标准化
# 1.对输入值标准化
# 求均值
for i in range(nInputNodes - 1):
    ss[i] = 0
    for j in range(nPatterns - 1):
        ss[i] = ss[i] + x[j][i]

    average[i] = ss[i] / nPatterns

# 求方差
for i in range(nInputNodes - 1):
    sss[i] = 0
    for j in range(nPatterns - 1):
        sss[i] = sss[i] + (x[j][i] - average[i]) * (x[j][i] - average[i])

    fangcha[i] = np.sqrt(sss[i] / (nPatterns - 1))

# 求自标准化后的输入值
for i in range(nInputNodes - 1):
    MIN = 1000.0
    MAX = -1000.0
    for j in range(nPatterns - 1):
        out0[j][i] = (x[j][i] - average[i]) / fangcha[i]
        if out0[j][i] <= MIN:
            MIN = out0[j][i]

        if out0[j][i] >= MAX:
            MAX = out0[j][i]

    out0[nPatterns][i] = MIN
    out0[nPatterns + 1][i] = MAX

# 标准化
for i in range(nInputNodes - 1):
    for j in range(nPatterns - 1):
        out0[j][i] = (out0[j][i] - out0[nPatterns][i]) / (out0[nPatterns + 1][i] - out0[nPatterns][i]) * 0.8 + 0.1

# 将标准化后的输入值输到数据文件中去
# ofstream my_file5
# my_file5.open("输入自标准化.dat", ios::out)
my_file5 = open("输入自标准化.dat", "w")
my_file5.write("")

# 将输入自标准化的权值写入数据文件
for i in range(nInputNodes - 1):
    my_file5.write(str(average[i]))
    my_file5.write(str(fangcha[i]))
    my_file5.write(str(out0[nPatterns][i]))
    my_file5.write(str(out0[nPatterns + 1][i]))

my_file5.close()

# 2.对输出值标准化
# 求均值
for i in range(nOutputNodes - 1):
    ssy[i] = 0
    for j in range(nPatterns - 1):
        ssy[i] = ssy[i] + y[j][i]
    averagey[i] = ssy[i] / nPatterns
    print("输出值平均值为:" + str(averagey[i]))

# 求方差
for i in range(nOutputNodes - 1):
    sssy[i] = 0
    for j in range(nPatterns - 1):
        sssy[i] = sssy[i] + (y[j][i] - averagey[i]) * (y[j][i] - averagey[i])

    fangchay[i] = sqrt(sssy[i] / (nPatterns - 1))

# 求自标准化后的输出值
for i in range(nOutputNodes - 1):
    MIN = 1000.0
    MAX = -1000.0
    for j in range(nPatterns - 1):
        target[j][i] = (y[j][i] - averagey[i]) / fangchay[i]
        if target[j][i] <= MIN:
            MIN = target[j][i]
        if target[j][i] >= MAX:
            MAX = target[j][i]
    target[nPatterns][i] = MIN
    target[nPatterns + 1][i] = MAX

# 标准化
for i in range(nOutputNodes - 1):
    for j in range(nPatterns - 1):
        target[j][i] = (target[j][i] - target[nPatterns][i]) / (
                target[nPatterns + 1][i] - target[nPatterns][i]) * 0.8 + 0.1

# 将自标准化后的输出值输到数据文件中去
# ofstream my_file6
# my_file6.open("输出自标准化.dat", ios::out)
my_file6 = open("输出自标准化.dat", "w")
my_file6.write("")

for i in range(nOutputNodes - 1):
    my_file6.write(str(averagey[i]))
    my_file6.write(str(fangchay[i]))
    my_file6.write(str(target[nPatterns][i]))
    my_file6.write(str(target[nPatterns + 1][i]))

my_file6.close()

# 输入 - 隐含层权值初始化
for i in range(nHiddenNodes):
    for j in range(nInputNodes):
        frand = random.randint(0, scaleValue)
        w1[i][j] = wmax * (1.0 - 2 * frand / scale)
        delw1[i][j] = 0.0

# 隐含-输出层的权值初始化
for i in range(nOutputNodes - 1):
    for j in range(nHiddenNodes):
        frand = random.randint(0, scaleValue)
        w2[i][j] = wmax * (1.0 - 2 * frand / scale)
        delw2[i][j] = 0.0

# 开始迭代循环
print("正在计算......")
for qq in range(nIterations):
    t1_start = perf_counter()
    if qq % 1000 == 0 and qq > 0:
        t1_stop = perf_counter()
        print('epoch ' + str(qq) + ' time: ' + str(t1_start - t1_stop))
    for p in range(nPatterns - 1):
        # 隐含层的输出值
        for h in range(nHiddenNodes - 1):
            SUM = w1[h][nInputNodes]
            for i in range(nInputNodes - 1):
                SUM = SUM + w1[h][i] * out0[p][i]

            out1[p][h] = 1.0 / (1.0 + exp(-SUM))

    # 输出层的输出值
    for j in range(nOutputNodes - 1):
        SUM = w2[j][nHiddenNodes]
        for h in range(nHiddenNodes - 1):
            SUM = SUM + w2[j][h] * out1[p][h]

        out2[p][j] = 1.0 / (1.0 + exp(-SUM))

    # 输出层的误差
    for j in range(nOutputNodes - 1):
        delta2[p][j] = (target[p][j] - out2[p][j]) * out2[p][j] * (1.0 - out2[p][j])

    # 隐含层的误差
    for h in range(nHiddenNodes - 1):
        SUM = 0.0
        for j in range(nOutputNodes - 1):
            SUM = SUM + delta2[p][j] * w2[j][h]

        delta1[p][h] = SUM * out1[p][h] * (1.0 - out1[p][h])

    # 调整隐含-输出层间的权值
    for j in range(nOutputNodes - 1):
        # dw = 0
        SUM = 0.0
        for p in range(nPatterns - 1):
            SUM = SUM + delta2[p][j]

        dw = eta * SUM + alpha * delw2[j][nHiddenNodes]
        w2[j][nHiddenNodes] += dw
        delw2[j][nHiddenNodes] = dw
        for h in range(nHiddenNodes - 1):
            SUM = 0.0
            for p in range(nPatterns - 1):
                SUM = SUM + delta2[p][j] * out1[p][h]

        dw = eta * SUM + alpha * delw2[j][h]
        w2[j][h] = w2[j][h] + dw
        delw2[j][h] = dw

    # 调整输入-隐含层的权值
    for h in range(nHiddenNodes - 1):
        # dw
        SUM = 0.0
        for p in range(nPatterns - 1):
            SUM = SUM + delta1[p][h]

        dw = eta * SUM + alpha * delw1[h][nInputNodes]
        w1[h][nInputNodes] += dw
        delw1[h][nInputNodes] = dw

        for i in range(nInputNodes - 1):
            SUM = 0.0
            for p in range(nPatterns - 1):
                SUM = SUM + delta1[p][h] * out0[p][i]

            dw = eta * SUM + alpha * delw1[h][i]
            w1[h][i] = w1[h][i] + dw
            delw1[h][i] = dw

    # 计算均方根误差
    if MonitorError or (qq % nReportErrors == 0):
        error1 = 0.0
        for p in range(nPatterns - 1):
            for j in range(nOutputNodes - 1):
                temp = target[p][j] - out2[p][j]
                error1 = error1 + temp * temp

        # 所有模式每个节点的平均误差
        error1 = error1 / (nPatterns * nOutputNodes)
        MonitorError = 0
        if error1 < errorLimit:
            break

print("标准化后的平均误差限为: " + str(error1))
# 迭代循环结束

# 将调节参数写入数据文件中去

my_file7 = open("调节参数.dat", "a")
my_file7.write("模式个数:" + str(nPatterns))
my_file7.write("输入节点个数:" + str(nInputNodes))
my_file7.write("输出层节点个数:" + str(nOutputNodes))
my_file7.write("隐含层节点个数:" + str(nHiddenNodes))
my_file7.write("最大权值:" + str(wmax))
my_file7.write("学习速率:" + str(eta))
my_file7.write("势态因子:" + str(alpha))
my_file7.write("误差平方限为:" + str(errorLimit))
my_file7.write("最终误差平方之和为:" + str(error1))
my_file7.write("迭代次数为:" + str(qq) + "\n")
my_file7.close()

# 将所有模式的目标值和输出值还原
for i in range(nOutputNodes - 1):
    for j in range(nPatterns - 1):
        target[j][i] = ((target[j][i] - 0.1) / 0.8 * (target[nPatterns + 1][i] - target[nPatterns][i]) +
                        target[nPatterns][i]) * fangchay[i] + averagey[i]
        out2[j][i] = ((out2[j][i] - 0.1) / 0.8 * (target[nPatterns + 1][i] - target[nPatterns][i]) + target[nPatterns][
            i]) * fangchay[i] + averagey[i]

# 将原始渣成分, 神经网络计算出来的渣成分, 相对误差输入数据文件
my_file = open("鼓风优化参数.dat", "w")
# my_file.write("")

error1 = 0.0
for p in range(nPatterns - 1):
    my_file = open("鼓风优化参数.dat", "a")
    for j in range(nOutputNodes - 1):
        my_file.write(str(p) + " " + str(out2[p][j]) + " " + str(target[p][j]) + " "
                      + str((target[p][j] - out2[p][j]) / target[p][j] * 100))
    temp = target[p][j] - out2[p][j]
    error1 = error1 + temp * temp

    my_file.close()

    error1 = error1 / (nPatterns * nOutputNodes)

    # 将权值写入数据文件

    # my_file2 = open("权值.dat", "r")
    # my_file2.write("")
    my_file2 = open("权值.dat", "w")
    for j in range(nInputNodes):
        my_file2.write("0")

    my_file2.close()
    # 将输入-隐含层的权值写入数据文件
    for i in range(nHiddenNodes):
        my_file2 = open("权值.dat", "a")
        for j in range(nInputNodes):
            # my_file2 << i << " " << j << " " << w1[i][j] << "\n"
            my_file2.write(str(w1[i][j]))

        my_file2.close()

    # 将隐含-输出层的权值写入数据文件
    for i in range(nOutputNodes - 1):
        my_file2 = open("权值.dat", "a")
        for j in range(nHiddenNodes):
            # my_file2 << i + " " + j + " " + w2[i][j] + "\n"
            my_file2.write(str(w2[i][j]))

        my_file2.close()

    print("训练已经结束")
    print("循环次数为:" + str(qq))
    print("最终原始平均误差为:" + str(error1))
