import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# --- Configurações da Animação ---
NUM_FRAMES = 100
ROTATION_ANGLE = 2 * np.pi

# --- Preparação do Gráfico ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


# --- Função de Simulação ---
def get_statevector_for_angle(angle):
    """Cria um circuito e retorna o statevector para um ângulo de rotação específico."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(angle, 0)
    qc.save_statevector()
    simulator = AerSimulator()
    result = simulator.run(qc).result()
    return result.get_statevector().data


# --- Função Principal da Animação ---
trajectory = []

def update(frame):
    """Esta função é chamada para cada quadro da animação."""
    ax.cla()  # Limpa o eixo para o próximo quadro

    current_angle = (frame / NUM_FRAMES) * ROTATION_ANGLE
    statevector = get_statevector_for_angle(current_angle)
    alpha = statevector[0]
    beta = statevector[1]
    theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1.0))
    phi = np.angle(beta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Calcula a trajetória do vetor de estado
    trajectory.append([x, y, z])
    traj_arr = np.array(trajectory)
    if frame == NUM_FRAMES - 1:
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], color='orange', linewidth=2, alpha=0.7)
    else:
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], color='orange', linewidth=2, alpha=0.7)

    # Projeções do vetor nos planos XY, XZ e YZ
    ax.plot([x, x], [y, y], [0, z], color='gray', linestyle='dashed', linewidth=1)
    ax.plot([x, 0], [y, 0], [0, 0], color='gray', linestyle='dashed', linewidth=1)
    ax.plot([x, x], [0, y], [z, 0], color='gray', linestyle='dashed', linewidth=1)
    ax.plot([x, 0], [0, 0], [z, 0], color='gray', linestyle='dashed', linewidth=1)
    ax.plot([0, x], [y, y], [z, 0], color='gray', linestyle='dashed', linewidth=1)
    ax.plot([0, 0], [y, 0], [z, 0], color='gray', linestyle='dashed', linewidth=1)

    # --- Redesenha tudo para o quadro atual ---
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    sphere_x = np.cos(u) * np.sin(v)
    sphere_y = np.sin(u) * np.sin(v)
    sphere_z = np.cos(v)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='c', alpha=0.15, linewidth=0)

    phase_color = plt.cm.hsv((phi % (2 * np.pi)) / (2 * np.pi))

    ax.quiver(0, 0, 0, x, y, z, length=1.0, color=phase_color, arrow_length_ratio=0.1, linewidth=3)

    ax.set_xlim([-1, 1]);
    ax.set_ylim([-1, 1]);
    ax.set_zlim([-1, 1])
    ax.set_aspect('equal')
    ax.set_title(f"Animação de Rotação de Fase (Ângulo φ = {np.rad2deg(phi):.1f}°)")


# --- Montagem e Execução da Animação ---
# O 'blit=False' também é importante para garantir a compatibilidade
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=50, blit=False)

plt.show()