#卡尔曼滤波器确定滚轮转速和施肥速率的关系
#dt不固定，在1上下波动
#基础卡尔曼滤波器，v为已知量（控制量）
#delta_m = alpha0+alpha1*v+alpha2*v**2
#alpha值为变化量
#真实值仿真方程与状态方程不同
import numpy as np
import matplotlib.pyplot as plt
import random

# 时间长度
N = 4000
t = [0]
for i in range(1, N):
    t.append(t[-1] + random.uniform(0.9, 1.1))


#定义估计量
# def v_est(t):
#     if 500 < t < 1500:
#         return 10
#     elif 2000 < t < 3000:
#         return 15
#     else:
#         return 0

#v任意变化
def v_est(t):
    if 1000 < t < 3000:
        return 20 * abs(np.sin(t / (50 * np.pi)))
    else:
        return 0

v = np.array([v_est(ti) for ti in t], dtype=np.float64)


def alpha0(t):
        return (-0.0005 * t + 2)

alpha0 = np.array([alpha0(ti) for ti in t], dtype=np.float64)

def alpha1(t):
    return (-0.00005 * t + 0.2)

alpha1 = np.array([alpha1(ti) for ti in t], dtype=np.float64)

def alpha2(t):
    return (-0.000005 * t + 0.02)

alpha2 = np.array([alpha2(ti) for ti in t], dtype=np.float64)

m_est = [20000]  #肥料初始重量

for k in range(1, N):
    m_k = m_est[k - 1] - (alpha0[k - 1] + alpha1[k - 1] * v[k - 1] + alpha2[k - 1] * v[k - 1] ** 2) * (t[k] - t[k-1])
    if m_k < 0:
        m_k = 0
    m_est.append(m_k)
    # delta_m_k = alpha0[k - 1] + alpha1[k - 1] * v[k - 1] + alpha2[k - 1] * v[k - 1] ** 2
    # delta_m_est.append(delta_m_k)

#计算delta_m
delta_est = np.diff(m_est)
delta_m_0 = 0
delta_m_est = np.insert(delta_est, 0, delta_m_0)

delta_m = [-x for x in delta_m_est]

#设置高斯白噪声
m_noise = np.random.normal(0, 0.03, N) #均值为0，标准差为0.03
m_mea = m_est + m_noise * m_est    #测量重量值

#卡尔曼滤波

X = np.mat([[20000], [2]])

P = np.mat([[1, 0], [0, 1]])  # 定义初始状态协方差矩阵
Q = np.mat([[0.01, 0], [0, 0.0001]])  # 定义状态转移(预测噪声)协方差矩阵
H = np.mat([1, 0])  # 定义观测矩阵
R = np.mat([[100000]])  # 定义观测噪声协方差

X_mat = np.zeros(N)
X_mat[0] = 20000
alpha_0mat = np.zeros(N)
alpha_0mat[0] = 2

k = 0  # 采样点计数
for i in range(N-1):
    k += 1
    A = np.mat([[1, -(t[k] - t[k - 1])], [0, 1]])  # 定义状态转移矩
    X_predict = A * X
    XX = X_predict.reshape(-1, 1)
    P_predict = A * P * A.T + Q
    K = P_predict * H.T / (H * P_predict * H.T + R)
    X = XX + K * (m_mea[k - 1] - H * XX)
    P = (np.eye(2) - K * H) @ P_predict
    X_mat[i + 1] = X[0, 0]
    alpha_0mat[i + 1] = X[1, 0]


#计算delta_m
delta_X = np.diff(X_mat)
delta_X_mat = np.insert(delta_X, 0, delta_m_0)

#滤波值与理想值的差值
error = []
for i in range(N):
    error_k = X_mat[i] - m_est[i]
    error.append(error_k)

plt.plot(t, error)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
ax1.plot(t, m_mea, 'b')
ax1.plot(t, X_mat, 'r')
ax1.plot(t, m_est, 'k')
ax1.set_xlabel('时间: t')
ax1.set_ylabel('质量: m')
ax1.legend(['滤波前', '滤波后', '理想值'])

ax2.plot(t, alpha_0mat, 'r')
# ax2.plot(t, delta_m, 'b')
ax2.set_xlabel('时间: t')
ax2.set_ylabel('系数：alpha0')
ax2.legend(['alpha0'])


ax3.plot(t, delta_X_mat, 'r')
ax3.plot(t, delta_m_est, 'b')
ax3.set_xlabel('时间: t')
ax3.set_ylabel('delta_m')
ax3.legend(['滤波值', '理想值'])

plt.show()
