"""
LASSO回归算法实现
包含次梯度法、临近点梯度法和ADMM算法
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局测试次数
testsize = 10


def QuestionGenerate(n, p):
    """
    问题生成函数
    生成测试数据A, x, b
    """
    A, x, b = [], [], []
    for i in range(testsize):
        np.random.seed(i)
        A.append((np.random.rand(n, p) - 0.5) * 10)
        x.append(np.random.randint(-10, 10, (p, 1)))
        b.append(np.dot(A[i], x[i]))
    return A, x, b


def SubGradMethod(A, b, Lambda):
    """
    次梯度法求解LASSO问题
    """
    Accuracy, Time, F = [], [], []
    for i in range(10):
        p = len(A[0][0])
        Accuracy.append([])
        Time.append([])
        F.append([])

        # 第一阶段：找到最优值f_star
        time, x1, f_star = 0, np.full((p, 1), 0), 10000000
        Miu = max(np.linalg.eig(np.dot(A[i], A[i].T))[0])
        A1, b1 = A[i], b[i]

        while time < 40000:
            time += 1
            # 计算次梯度
            SubGrad1 = np.full((p, 1), 0)
            for j in range(p):
                if x1[j][0] > 0:
                    SubGrad1[j][0] = 1
                elif x1[j][0] < -0:
                    SubGrad1[j][0] = -1
                else:
                    SubGrad1[j][0] = 0

            SubGrad2 = np.dot(A1.T, np.dot(A1, x1) - b1) + Lambda * SubGrad1
            # 更新x
            x1 = x1 - (1 / (time + Miu)) * SubGrad2
            # 计算目标函数值
            f = 0.5 * np.linalg.norm(np.dot(A1, x1) - b1, ord=2, axis=None, keepdims=False) + Lambda * np.linalg.norm(
                x1, ord=1, axis=None, keepdims=False)
            if f < f_star:
                f_star = f

        # 第二阶段：记录迭代过程
        time, x1, change = 0, np.full((p, 1), 0.01), 10000
        while time < 10000:
            time += 1
            SubGrad1 = np.full((p, 1), 0)
            for j in range(p):
                if x1[j][0] > 0.0001:
                    SubGrad1[j][0] = 1
                elif x1[j][0] < -0.0001:
                    SubGrad1[j][0] = -1
                else:
                    SubGrad1[j][0] = 0

            SubGrad2 = np.dot(A1.T, np.dot(A1, x1) - b1) + Lambda * SubGrad1
            x1 = x1 - (1 / (time + Miu)) * SubGrad2
            f = 0.5 * np.linalg.norm(np.dot(A1, x1) - b1, ord=2, axis=None, keepdims=False) + Lambda * np.linalg.norm(
                x1, ord=1, axis=None, keepdims=False)
            F[i].append(f)

            if f == f_star:
                Accuracy[i].append(Accuracy[i][time - 2] - 3)
            else:
                Accuracy[i].append(np.log10((f - f_star) / f_star))
            Time[i].append(time)

    return Accuracy, Time, F


def ProximalGradMethod(A, b, Lambda):
    """
    临近点梯度法求解LASSO问题
    """
    Accuracy, Time, F = [], [], []
    for i in range(10):
        p = len(A[0][0])
        Accuracy.append([])
        Time.append([])
        F.append([])

        # 第一阶段：找到最优值f_star
        time, x1, f_star = 0, np.full((p, 1), 0), 1000000
        Miu = 1 / max(np.linalg.eig(np.dot(A[i], A[i].T))[0])
        A1, b1 = A[i], b[i]

        while time < 40000:
            time += 1
            Grad = np.dot(A1.T, np.dot(A1, x1) - b1)
            y1 = x1 - Miu * Grad
            # 软阈值操作
            x1_star = np.sign(y1) * np.maximum(np.abs(y1) - Lambda * Miu, 0)
            x1 = x1_star
            f = np.linalg.norm(0.5 * (np.dot(A1, x1) - b1), ord=2, axis=None, keepdims=False) + Lambda * np.linalg.norm(
                x1, ord=1, axis=None, keepdims=False)
            if f < f_star:
                f_star = f

        # 第二阶段：记录迭代过程
        time, x1 = 0, np.full((p, 1), 0)
        while time < 10000:
            time += 1
            Grad = np.dot(A1.T, np.dot(A1, x1) - b1)
            y1 = x1 - Miu * Grad
            x1_star = np.sign(y1) * np.maximum(np.abs(y1) - Lambda * Miu, 0)
            x1 = x1_star
            f = np.linalg.norm(0.5 * (np.dot(A1, x1) - b1), ord=2, axis=None, keepdims=False) + Lambda * np.linalg.norm(
                x1, ord=1, axis=None, keepdims=False)
            F[i].append(f)

            if f == f_star:
                Accuracy[i].append(Accuracy[i][time - 2] - 3)
            else:
                Accuracy[i].append(np.log10((f - f_star) / f_star))
            Time[i].append(time)

    return Accuracy, Time, F


def ADMM(A, b, Lambda):
    """
    ADMM算法求解LASSO问题
    """
    Accuracy, Time, F, Rou = [], [], [], 0.1
    for i in range(10):
        p = len(A[0][0])
        Accuracy.append([])
        Time.append([])
        F.append([])

        # 第一阶段：找到最优值f_star
        time, x1, z, y, f_star = 0, np.full((p, 1), 0), np.full((p, 1), 0), np.full((p, 1), 0), 1000000
        A1, b1 = A[i], b[i]
        Miu = 1 / max(np.linalg.eig(np.dot(A1, A1.T))[0])
        Rev = np.linalg.inv(np.dot(A1.T, A1) + Rou * np.eye(p))

        while time < 40000:
            f = np.linalg.norm(0.5 * (np.dot(A1, x1) - b1), ord=2, axis=None, keepdims=False) + Lambda * np.linalg.norm(
                x1, ord=1, axis=None, keepdims=False)
            F[i].append(f)
            time += 1
            # ADMM更新步骤
            x1 = np.dot(Rev, np.dot(A1.T, b1) + Rou * (z - y))
            z = np.sign(x1 + Rou * y) * np.maximum(np.abs(x1 + Rou * y) - Miu / Rou, 0)
            y = y + x1 - z

            if f < f_star:
                f_star = f

        # 第二阶段：记录迭代过程
        time, x1, z, y = 0, np.full((p, 1), 0), np.full((p, 1), 0), np.full((p, 1), 0)
        while time < 10000:
            f = np.linalg.norm(0.5 * (np.dot(A1, x1) - b1), ord=2, axis=None, keepdims=False) + Lambda * np.linalg.norm(
                x1, ord=1, axis=None, keepdims=False)
            time += 1
            x1 = np.dot(Rev, np.dot(A1.T, b1) + Rou * (z - y))
            z = np.sign(x1 + Rou * y) * np.maximum(np.abs(x1 + Rou * y) - Miu / Rou, 0)
            y = y + x1 - z

            if f == f_star:
                Accuracy[i].append(Accuracy[i][time - 2] - 3)
            else:
                Accuracy[i].append(np.log10((f - f_star) / f_star))
            Time[i].append(time)

    return Accuracy, Time, F


def plotADMM(Accuracy, Time, F, n, p):
    """
    可视化ADMM算法的迭代结果
    """
    # 绘制精度图
    plt.figure(figsize=(48, 20))
    plt.title(f"对于n = {n}, p = {p}的Lasso问题的ADMM迭代结果:")
    plt.ylabel("log10(f^k-f*)/f*")
    plt.xlabel("k")

    for i in range(testsize):
        # 生成渐变色
        color_val = 25 * i + 25
        hex_str = format(color_val, '02x')
        plt.plot(Time[i][:99], Accuracy[i][:99], f"#00{hex_str}{hex_str}")

    # 绘制目标函数值图
    plt.figure(figsize=(24, 10))
    plt.title(f"对于n = {n}, p = {p}的Lasso问题的ADMM迭代结果:")
    plt.ylabel("||Ax^k-b||^2+λ|x|")
    plt.xlabel("k")

    for i in range(testsize):
        color_val = 25 * i + 25
        hex_str = format(color_val, '02x')
        plt.plot(Time[i][:100], F[i][:100], f"#00{hex_str}{hex_str}")


def plot(Accuracy, Time, F, n, p, Method):
    """
    通用绘图函数
    """
    # 绘制精度图
    plt.figure(figsize=(48, 20))
    plt.title(f"对于n = {n}, p = {p}的Lasso问题的{Method}迭代结果:")
    plt.ylabel("log10(f^k-f*)/f*")
    plt.xlabel("k")

    for i in range(testsize):
        color_val = 25 * i + 25
        hex_str = format(color_val, '02x')
        plt.plot(Time[i], Accuracy[i], f"#00{hex_str}{hex_str}")

    # 绘制目标函数值图
    plt.figure(figsize=(24, 10))
    plt.title(f"对于n = {n}, p = {p}的Lasso问题的{Method}迭代结果:")
    plt.ylabel("||Ax^k-b||^2+λ|x|")
    plt.xlabel("k")

    for i in range(testsize):
        color_val = 25 * i + 25
        hex_str = format(color_val, '02x')
        plt.plot(Time[i], F[i], f"#00{hex_str}{hex_str}")


def main():
    """
    主函数，运行不同配置的LASSO问题
    """
    print("图像会在运行结束后一起出现")

    # 配置列表：[(n, p), ...]
    configs = [(10, 5), (100, 5), (10, 50), (10, 500)]
    Lambda = 1  # 正则化参数

    for n, p in configs:
        print(f"\n对于n = {n}, p = {p}的Lasso问题的次梯度法迭代结果:")
        A, x, b = QuestionGenerate(n, p)
        Accuracy_SG, Time_SG, F_SG = SubGradMethod(A, b, Lambda)
        plot(Accuracy_SG, Time_SG, F_SG, n, p, "次梯度法")

        print(f"对于n = {n}, p = {p}的Lasso问题的临近点梯度法迭代结果:")
        Accuracy_PG, Time_PG, F_PG = ProximalGradMethod(A, b, Lambda)
        plot(Accuracy_PG, Time_PG, F_PG, n, p, "临近点梯度法")

        print(f"对于n = {n}, p = {p}的Lasso问题的ADMM迭代结果:")
        Accuracy_ADMM, Time_ADMM, F_ADMM = ADMM(A, b, Lambda)
        plotADMM(Accuracy_ADMM, Time_ADMM, F_ADMM, n, p)

        print("图像会在运行结束后一起出现")

    # 显示所有图像
    plt.show()


if __name__ == "__main__":
    main()