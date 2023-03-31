import numpy as np
import matplotlib.pyplot as plt

'''def get_cov_diff(dim):
    mean = np.zeros([dim])  # 均值向量
    matrix = np.eye(dim)
    # using np.multinomial() method
    d = np.random.multivariate_normal(mean, matrix, 100)  # 5生成5个样本点
    cov = np.cov(d.T)
    diff = np.linalg.norm(matrix - cov) / (np.linalg.norm(matrix))
    return diff


l = []
for i in range(1, 201):
    l.append(get_cov_diff(i))

plt.figure(dpi=150)
plt.xlabel("d/n")
plt.ylabel("relative difference")
plt.plot(np.linspace(0.1,2,200),l)
plt.grid()
plt.tight_layout()
plt.savefig('./imgs/covariance.png')'''

mean = np.zeros([2000])  # 均值向量
matrix = np.eye(2000)
# using np.multinomial() method
gfg = np.random.multivariate_normal(mean, matrix, 4000)  # 5生成5个样本点

cov = np.cov(gfg.T)
_, v, _ = np.linalg.svd(cov)
v = v[np.argwhere(v > 1e-9)]

plt.figure(dpi=200)
plt.hist(v, bins=50)

plt.xlabel("λ")
plt.ylabel("hist")
plt.tight_layout()
plt.savefig('特征值.png')
