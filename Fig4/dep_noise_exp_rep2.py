import numpy as np
from tqdm import tqdm
import os
import math
from itertools import combinations
from ibmn import ibm_n

script_dir = os.path.dirname(os.path.abspath(__file__))
rep = (1,1)


n_shots_channel = 2**6
num_samples = 2**8
n_shots_measurement = 2**18




def draw_expressibility(results, p_list, lam_list, save_path=None):
    tar_lams = lam_list

    # results = np.load(os.path.join(script_dir, f'mitigation_data_seed{42}_lam{lam}_num_samples{num_samples}.npy'))


    #noise_less_results = np.load(os.path.join(script_dir, f'exp_data_q{rep}_fun{"noise_expressibility_lam"}_samples{2**16}_shots{2**12}_channel{1}.npy'))
    noise_less_results = np.load(os.path.join(script_dir, f'exp_data_q{rep}_fun{"noise_expressibility_lam"}_samples{2**15}_shots{2**15}_channel{1}.npy'))

    #results=(fun_list, p_list, seeds)

    # 对采样结果计算均值和标准差
    mean_results = results.mean(axis=2)
    std_results = results.std(axis=2)

    mean_noise_less_results = noise_less_results.mean(axis=1)
    std_noise_less_results = noise_less_results.std(axis=1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))

    cmap = plt.cm.tab10  # Using reversed YlOrRd colormap
    colors = [cmap(i) for i in (range(len(p_list)))]

    for i, p in enumerate(p_list):
        mean_line = np.concatenate([[mean_noise_less_results[i]],mean_results[:,i]])
        std_line = np.concatenate([[std_noise_less_results[i]],std_results[:,i]])
        
        plt.plot(lam_list, mean_line, '-', color=colors[i],
         label=f"Depth={p*4}", linewidth=2)

        # 绘制除第一个点外的所有圆形标记
        plt.plot(lam_list[1:], mean_line[1:], '^', color=colors[i], markersize=6)

        # 单独绘制第一个点为三角形
        plt.plot(lam_list[0], mean_line[0], 'o', color=colors[i],markersize=6)

        plt.fill_between(lam_list, mean_line - std_line, mean_line + std_line, color=colors[i], alpha=0.3)

    # plt.plot(positions, np.ones(len(positions))*mean_results[0][0], '--', color='black', label=f'λ = {lam}', linewidth=1)

    plt.yscale('log')
    # Set custom y-axis ticks at specific powers of 10
    # yticks = [2e-5,5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
    # plt.yticks(yticks, [f'{y:.1e}' for y in yticks])

    # Add major and minor gridlines
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle='--', alpha=0.2)
    plt.minorticks_on()

    plt.xlabel('$\gamma$ (Damping Parameter)')
    plt.ylabel('Deviation')
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # 创建自定义图例条目
    from matplotlib.lines import Line2D
    custom_handles = [
        Line2D([0], [0], marker='o', color='black', label="$\widetilde{\mathcal{M}}_2^2$", 
               markersize=6, linestyle='None'),
        Line2D([0], [0], marker='^', color='black', label="$\widetilde{\mathcal{M}}_{2\leq}^2$", 
               markersize=6, linestyle='None')
    ]
    
    # 合并现有图例和自定义条目
    all_handles = handles + custom_handles
    all_labels = labels + ["$\widetilde{\mathcal{M}}_2^2$", "$\widetilde{\mathcal{M}}_{2\leq}^2$"]
    
    plt.legend(all_handles, all_labels, reverse=False, loc='lower right')
    
    #plt.legend(reverse=False)
    # plt.title('Noise Mitigation Performance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to {save_path}")

    plt.show()
    # draw_mitigation(res,num_samples = int(2**18),size=100, save_path='bem.pdf')


if __name__ == '__main__':
    print(f'exp_q{rep}_samples{num_samples}_shots{n_shots_measurement}_channel{n_shots_channel}')
    size = 100
    np.set_printoptions(threshold=np.inf)
    seed_init = 42
    np.random.seed(seed_init)
    seeds = np.random.randint(0, 100000, size=size)
    seeds[0] = 42

    p_list = [1, 2, 3]
    lam_list = [#"noise_expressibility_lam",
                #"noise_expressibility_lam_lowerbound(0.0)",
                "noise_expressibility_lam_lowerbound(0.05)", 
                "noise_expressibility_lam_lowerbound(0.1)", 
                "noise_expressibility_lam_lowerbound(0.15)", 
                "noise_expressibility_lam_lowerbound(0.2)",
                "noise_expressibility_lam_lowerbound(0.3)",
                "noise_expressibility_lam_lowerbound(0.4)",
                "noise_expressibility_lam_lowerbound(0.5)",
                #"noise_expressibility_lam_lowerbound_norm1(0.3)"
                ]

    result = np.zeros((len(lam_list), len(p_list), size))

    total_iterations = len(lam_list) * len(p_list) * len(seeds)
    with tqdm(total=total_iterations, desc="total_iterations") as pbar:
        for i, fun in enumerate(lam_list):
            if os.path.exists(os.path.join(script_dir, f'exp_data_q{rep}_fun{lam_list[i]}_samples{num_samples}_shots{n_shots_measurement}_channel{n_shots_channel}.npy')):
                result[i] = np.load(os.path.join(script_dir, f'exp_data_q{rep}_fun{lam_list[i]}_samples{num_samples}_shots{n_shots_measurement}_channel{n_shots_channel}.npy'))
                pbar.update(len(p_list) * len(seeds))

    #result = np.load(os.path.join(script_dir, f'exp_data_q{rep}_samples{num_samples}_shots{n_shots_measurement}_channel{n_shots_channel}.npy'))
    lam_list = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    draw_expressibility(results=result, p_list=p_list, lam_list=lam_list, save_path=os.path.join(script_dir, f'exp_fig_q{rep}_samples{num_samples}_shots{n_shots_measurement}_channel{n_shots_channel}.pdf'))
