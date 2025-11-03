import numpy as np
from tqdm import tqdm
from ibm435 import qubits,two_qubit_gates, qubits_dict,qubits_dict_inv, gate_groups,plot_lattice

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
p=15
lam=0.007
obs=[(12,13)]#,(12,14)

                    
def draw_mitigation(sensitivity,num_samples = int(1e4),size=10, save_path=None):
    tar_lams = [0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005]
    positions = list(range(0, 12))

    results = np.load(os.path.join(script_dir, f'mitigation_data_seed{42}_lam{lam}_num_samples{num_samples}.npy'))

    # 对采样结果计算均值和标准差
    mean_results = results.mean(axis=2)
    std_results = results.std(axis=2)


    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))

    cmap = plt.cm.YlOrRd_r  # Using reversed YlOrRd colormap
    colors = [cmap(i/len(tar_lams)) for i in reversed(range(len(tar_lams)))]

    for i, tar_lam in enumerate(tar_lams):
        mean_line = mean_results[i]
        std_line = std_results[i]
        plt.plot(positions, mean_line, '-o', color=colors[i],
                 label=f'λ = {tar_lam}', linewidth=2, markersize=6)
        plt.fill_between(positions, mean_line - std_line, mean_line + std_line, color=colors[i], alpha=0.3)
    
    plt.plot(positions[7], mean_results[3][7], '*',
                color=colors[3], markersize=15, 
                markeredgecolor='black', markeredgewidth=0.5)
    plt.plot(positions, np.ones(len(positions))*mean_results[0][0], '--', color='black',
                 label=f'λ = {lam}', linewidth=1)

    plt.yscale('log')
    # Set custom y-axis ticks at specific powers of 10
    yticks = [2e-5,5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
    plt.yticks(yticks, [f'{y:.1e}' for y in yticks])

    # Add major and minor gridlines
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.2)
    plt.minorticks_on()

    plt.xlabel('Mitigated Qubits')
    plt.ylabel('MSE')
    plt.legend(reverse=True)
    #plt.title('Noise Mitigation Performance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to {save_path}")

    plt.show()
    #draw_mitigation(res,num_samples = int(2**18),size=100, save_path='bem.pdf')

if __name__ == '__main__':
    
    num_samples_main = int(2**22)
    seed_init_main = 42
    data_filename = os.path.join(script_dir, f'noise_mse_qubits_seed{seed_init_main}_lam{lam}_num_samples{num_samples_main}.npy')
    res = np.load(data_filename)

    #res= res_test
    res = np.array(res)
    non_zero_min = res[res > 0].min()
    res_plus = np.array(res)+1*non_zero_min
    smooth_res=np.log(res_plus)
    plot_lattice(smooth_res, save_path='chip435.pdf')#save_path='chip435.pdf'

    draw_mitigation(res,num_samples = int(2**18),size=100, save_path='bem.pdf')