import matplotlib.pyplot as plt
import matplotlib
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


def show_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def show_policy_black(V):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())
    print(min_x, min_y, max_x, max_y)

    data = np.zeros( (max_x+1, max_y+1), dtype=np.float32)

    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            state = tuple([i,j,False])
            data[i][j] =V[state]
    fig, ax = plt.subplots(figsize=(max_x+1, max_y+1))

    cax = ax.matshow(data)
    # 显示图像
    plt.show()