import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

np.set_printoptions(suppress=True)
# 设置全局中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


def show_grid(env, data=None):
    fig, ax = plt.subplots(figsize=(env.ncol, env.nrow))
    # 设置颜色
    colors = [(0, 'red'), (0.5, 'white'), (1, 'yellow')]
    mycmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors)
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    cax = ax.matshow(env.grid, cmap=mycmap, norm=norm)

    data = data.reshape(env.nrow, env.ncol)

    if data is not None:
        for (i, j), val in np.ndenumerate(data):
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')
    # 设置网格线
    ax.set_xticks(np.arange(-0.5, env.nrow, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.ncol, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # 显示图像
    plt.show()


def show_policy(env, policy):
    fig, ax = plt.subplots(figsize=(env.ncol, env.nrow))
    # 设置颜色
    colors = [(0, 'red'), (0.5, 'white'), (1, 'yellow')]
    mycmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors)
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    cax = ax.matshow(env.grid, cmap=mycmap, norm=norm)

    data = env.grid

    # 在每个单元格中添加文本
    for (i, j), val in np.ndenumerate(data):
        teval = '↓'
        for k in range(env.action_size):
            a_star = np.argmax(policy[i*env.nrow + j])
            if a_star == 0:
                teval = '←'
            elif a_star == 1:
                teval = '↓'
            elif a_star == 2:
                teval = '→'
            elif a_star == 3:
                teval = '↑'
        ax.text(j, i, f'{teval}', ha='center', va='center', color='black')

    # 设置网格线
    ax.set_xticks(np.arange(-0.5, env.nrow, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.ncol, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # 显示图像
    plt.show()