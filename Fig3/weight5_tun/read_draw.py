import numpy as np
from ibmn import ibm_n
#import mc2moment as mc
import logging,os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(script_dir, f'grad_variance_y_rep{6}_x_rep{6}_{timestamp}.log')

def read(p,lam,seed=42,y_rep=6,x_rep=6):
    path=os.path.join(script_dir, f'grad_variance_y_rep{y_rep}_x_rep{x_rep}_p{p}_lam{lam}_seed{seed}.npy')
    if os.path.exists(path):
        res_gate_wise = np.load(path)
        return res_gate_wise
    else:
        print(f"File {path} does not exist. Please run the simulation first.")




def qubit_wise_post_process(res_qubit_wise, min_eta=1e-10):
    res = np.array(res_qubit_wise)
    if len([res > 0])==0:
        non_zero_min = min_eta
    else:
        non_zero_min = res[res > 0].min()
    res_plus = np.array(res)+1e-2*non_zero_min
    smooth_res=np.log(res_plus)
    return smooth_res

def draw_ad_gradsum(grad_sum, lams, ps, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))

    cmap = plt.cm.YlOrRd_r  # Using reversed YlOrRd colormap
    colors = [cmap(i/len(lams)) for i in range(len(lams))]

    for i, tar_lam in enumerate(lams):
        plt.plot(ps, grad_sum[i], '-o', color=colors[i],
                 label=f'$\gamma = {tar_lam}$', linewidth=2, markersize=6)

    plt.yscale('log')
    # Set custom y-axis ticks at specific powers of 10
    #yticks = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
    #plt.yticks(yticks, [f'{y:.1e}' for y in yticks])

    # Add major and minor gridlines
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.2)
    plt.minorticks_on()

    plt.xlabel('Circuit Depth')
    plt.ylabel('MSE (log scale)')
    plt.legend(reverse=False)
    #plt.title('Noise Mitigation Performance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to {save_path}")

    plt.show()
#plot_lattice(smooth_res)#save_path='chip435.pdf'

def draw_plot(result, p_list, lam_list,save_path=None):
    import matplotlib.pyplot as plt
    # 计算均值和标准差
    mean_result = np.mean(result, axis=2)  # 对 k 维度求均值
    std_result = np.std(result, axis=2)   # 对 k 维度求标准差

    # 绘制图形
    plt.figure(figsize=(8, 5))
    for i, p in enumerate(p_list):
        mean = mean_result[i, :]
        std = std_result[i, :]
        plt.plot(lam_list, mean, label=f'Depth={p*5}', marker='o')  # , linewidth=2, markersize=6
        plt.fill_between(
            lam_list, 
            mean - std, 
            mean + std, 
            alpha=0.3, 
        )

    # Add major and minor gridlines
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.2)
    plt.minorticks_on()

    plt.yscale('log')
    plt.xlabel('$\gamma$ (Damping Parameter)')
    plt.ylabel('Variance of Gradient')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    size = 100
    np.set_printoptions(threshold=np.inf)
    seed_init = 42
    np.random.seed(seed_init)
    seeds=np.random.randint(0, 100000, size=size)
    seeds[0] = 42
    lams = [0, 0.05, 0.1 ,0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    ps=[1,2,3]
    res = []

    gradient_qubut_wise = np.zeros(435)

    for p in ps:
        res_ele = []
        for lam in lams:
            res_seed=[]
            for seed in seeds:
                res_gate_wise = read(p, lam,seed)
                try:
                    val=res_gate_wise.sum()
                    if p == 3 and lam == 0.25: gradient_qubut_wise += res_gate_wise.sum(axis=1)
                except Exception as e:
                    print(f"Error processing p={p}, lam={lam}, seed={seed}: {e}")
                res_seed.append(val)
            res_ele.append(res_seed)
        res.append(res_ele)
    res = np.array(res)
    draw_plot(res, ps, lams, save_path="ad_train.pdf")#save_path="ad_train.pdf"

    y_rep=6
    x_rep=6
    qubits, two_qubit_gates, qubits_dict, gate_groups, plot_lattice = ibm_n(y_rep=y_rep,x_rep=x_rep)

    non_zero_min = gradient_qubut_wise[gradient_qubut_wise > 0].min()
    res_plus = np.array(gradient_qubut_wise)+1e-2*non_zero_min
    smooth_res=np.log(res_plus)
    plot_lattice(smooth_res,save_path='ad_train_chip435.pdf')#save_path='ad_train_chip435.pdf'

