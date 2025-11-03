
x=27
y=25

per1=[[1,2,3],[5,6,7],[9,10,11],[13,14,15],[17,18,19],[21,22,23],[25]]
per2=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25]]
per3=[[1],[3,4,5],[7,8,9],[11,12,13],[15,16,17],[19,20,21],[23,24,25]]
per4=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25]]
per=[per1,per2,per3,per4]

qubits_index=[]
two_qubit_gates=[]
for i in range(x):
    ind=i % 4
    tar_per=per[ind]
    for pair in tar_per:
        for j in range(y):
            if j+1 in pair:
                qubits_index.append((i,j))
            if j+1 in pair and j+2 in pair:
                two_qubit_gates.append(((i,j),(i,j+1)))


for (a,b) in qubits_index:
    if (a+1,b) in qubits_index:
        two_qubit_gates.append(((a,b),(a+1,b)))

qubits=len(qubits_index)
qubits_dict = {qubit: i for i, qubit in enumerate(qubits_index)}
qubits_dict_inv = {v: k for k, v in qubits_dict.items()}


def plot_lattice(point_values=None ,gate_groups=None ,save_path=None):
    x_scale=1
    y_scale=1
    point_size=70
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(8, 6))


    # 创建自定义警告色彩映射
    group_colors = ['blue', 'green', 'red']  # Blue, Green, Orange
    colors = ['#4575B4', '#FFFFBF', '#D73027']  # 绿色 -> 黄色 -> 红色
    positions = [0, 0.2, 1.0]
    warning_cmap = LinearSegmentedColormap.from_list('warning', list(zip(positions, colors)))
    #warning_cmap = LinearSegmentedColormap.from_list('warning', colors)
    cmap = warning_cmap #'coolwarm'

    if point_values is None:
        point_values = np.zeros(len(qubits_index))
    # Plot qubit positions
    x_coords = [pos[0] * x_scale for pos in qubits_index]
    y_coords = [pos[1] * y_scale for pos in qubits_index]
    scatter = ax.scatter(y_coords, -np.array(x_coords), c=point_values, cmap=cmap, s=point_size, label='Qubits')

    if gate_groups==None:
        # Plot connections (two-qubit gates)
        for (a1, b1), (a2, b2) in two_qubit_gates:
            ax.plot([b1 * y_scale, b2 * y_scale],
                    [-a1 * x_scale, -a2 * x_scale],
                    'gray', alpha=0.5)
    else:
        for group_idx, group in enumerate(gate_groups):
            for (a1, b1), (a2, b2) in group:
                ax.plot([b1 * y_scale, b2 * y_scale],
                        [-a1 * x_scale, -a2 * x_scale],
                        color=group_colors[group_idx], alpha=0.7, linewidth=1.5)


    #ax.grid(True, linestyle='--', alpha=0.3)
    #ax.set_xlabel('Y axis')
    #ax.set_ylabel('X axis')
    #ax.set_title('Quantum Lattice Structure')
    #ax.legend()

    #plt.colorbar(scatter)
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.02, shrink=0.5)
    cbar.ax.tick_params(labelsize=8)  # 调整刻度字体大小

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',transparent=True)
        print(f"Image saved to {save_path}")

    plt.show()
def partition_two_qubit_gates(two_qubit_gates):
    # Create three groups for gates
    groups = [[], [], []]
    # Track used qubits for each group
    used_qubits = [set(), set(), set()]

    for gate in two_qubit_gates:
        (q1x, q1y), (q2x, q2y) = gate
        q1 = (q1x, q1y)
        q2 = (q2x, q2y)

        # Try to assign gate to a group where its qubits haven't been used
        assigned = False
        for group_idx in range(3):
            if q1 not in used_qubits[group_idx] and q2 not in used_qubits[group_idx]:
                groups[group_idx].append(gate)
                used_qubits[group_idx].add(q1)
                used_qubits[group_idx].add(q2)
                assigned = True
                break

        if not assigned:
            raise ValueError(f"Cannot assign gate {gate} to any group without reusing qubits.")

    return groups
gate_groups = partition_two_qubit_gates(two_qubit_gates)
if __name__ == '__main__':
    print("Qubits number:", len(qubits_index))

    print("\nTwo-qubit gates number:", len(two_qubit_gates))
    print("\nQubits dictionary:", qubits_dict)


    print(gate_groups)
    plot_lattice(gate_groups=gate_groups)
    print(len(gate_groups[0]), len(gate_groups[1]), len(gate_groups[2]))