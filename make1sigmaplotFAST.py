import numpy as np
import scipy as sp
import astropy as ap
import matplotlib.pyplot as plt
import matplotlib as mpl

FM_FULL_label = ["A", "b_HI", "Tb", "sigma_NL", "sigma8tot", "n_s", "f", "DA", "H", "omegak", "omegaDE", "w0", "wa", "h", "gamma", "sigma_8", "fs8", "bs8"]

def get_FISHER(survey,bin_number,label_index=list(range(18))):
    full_fisher = np.loadtxt("./output/%s-fisher-full-%d.dat"%(survey,bin_number))
    full_fisher = np.linalg.inv(full_fisher[label_index][:,label_index])
    fh = full_fisher
    return fh


def make_ellipses(mean, cov, ax, confidence=5.991, alpha=0.3, color="blue", eigv=False, arrow_color_list=None):
    """
    多元正态分布
    mean: 均值
    cov: 协方差矩阵
    ax: 画布的Axes对象
    confidence: 置信椭圆置信率 # 置信区间， 95%： 5.991  99%： 9.21  90%： 4.605 
    alpha: 椭圆透明度
    eigv: 是否画特征向量
    arrow_color_list: 箭头颜色列表
    """
    lambda_, v = np.linalg.eig(cov)    # 计算特征值lambda_和特征向量v
    # print "lambda: ", lambda_
    # print "v: ", v
    # print "v[0, 0]: ", v[0, 0]

    sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
    height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
    angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆

    ax.add_artist(ell)
    ell.set_alpha(alpha)
    # 是否画出特征向量
    if eigv:
        # print "type(v): ", type(v)
        if arrow_color_list is None:
            arrow_color_list = [color for i in range(v.shape[0])]
        for i in range(v.shape[0]):
            v_i = v[:, i]
            scale_variable = np.sqrt(s) * sqrt_lambda[i]
            ax.arrow(mean[0], mean[1], scale_variable*v_i[0], scale_variable * v_i[1], 
                     width=0.05, 
                     length_includes_head=True, 
                     head_width=0.2, 
                     head_length=0.3,
                     color=arrow_color_list[i])

def plot_2D_gaussian_sampling(mean, cov, ax, scatter_plot = False, data_num=100, confidence=5.991, color="blue", alpha=0.3, eigv=False):
    """
    mean: 均值
    cov: 协方差矩阵
    ax: Axes对象
    confidence: 置信椭圆的置信率
    data_num: 散点采样数量
    color: 颜色
    alpha: 透明度
    eigv: 是否画特征向量的箭头
    """
    if isinstance(mean, list) and len(mean) > 2:
        print("Multivariate normal distribution, more than 2 dimensions")
        mean = mean[:2]
        cov_temp = []
        for i in range(2):
            cov_temp.append(cov[i][:2])
        cov = cov_temp
    elif isinstance(mean, np.ndarray) and mean.shape[0] > 2:
        mean = mean[:2]
        cov = cov[:2, :2]
    make_ellipses(mean, cov, ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)
    if scatter_plot:
        data = np.random.multivariate_normal(mean, cov, data_num)
        x, y = data.T
        ax.scatter(x, y, s=10, c=color)
    


def main():
    # plt.figure("Multivariable Gaussian Distribution")
    # plt.rcParams["figure.figsize"] = (8.0, 8.0)
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([0,9])
    ax.set_ylim([0,9])
    print("ax:", ax)

    mean = [4, 0]
    cov = [[1, 0.9], 
           [0.9, 0.5]]
    
    plot_2D_gaussian_sampling(mean=mean, cov=cov, ax=ax, eigv=False, color="r")

    mean1 = [5, 2]
    cov1 = [[1, 0],
           [0, 1]]
    plot_2D_gaussian_sampling(mean=mean1, cov=cov1, ax=ax, eigv=False)

    plt.savefig("./figs/gaussian_covariance_matrix.png")
    plt.close()



if __name__ == "__main__":
    # main()
    test = get_FISHER("FAST_hrx_opt",0)
    print(test[12][12])
