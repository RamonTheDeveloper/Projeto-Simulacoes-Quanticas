import numpy as np
import matplotlib

matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def simulate_qubit_state(gate_name='h'):
    """Simula o estado de um único qubit após a aplicação de portas quânticas."""
    qc = QuantumCircuit(1)

    # Portas Básicas Originais
    if gate_name == 'h':
        qc.h(0) # Leva ao eixo +X (Estado |+>)
    elif gate_name == 'x':
        qc.x(0)
    elif gate_name == 'y':
        qc.y(0)
    elif gate_name == 'z':
        qc.z(0)
        
    # --- NOVAS PORTAS (Para alcançar os círculos verdes no equador) ---
    elif gate_name == '-x':
        qc.x(0) # Primeiro inverte para |1>
        qc.h(0) # Depois aplica Hadamard. Leva ao eixo -X (Estado |->)
    elif gate_name == '+y':
        qc.h(0) # Leva ao equador (+X)
        qc.s(0) # Gira 90 graus para chegar no eixo +Y (Estado |i>)
    elif gate_name == '-y':
        qc.h(0) # Leva ao equador (+X)
        qc.sdg(0) # Gira -90 graus para chegar no eixo -Y (Estado |-i>)

    print("--> Circuito Quântico Criado:")
    print(qc)

    qc.save_statevector()

    simulator = AerSimulator()
    result = simulator.run(qc).result()
    statevector = result.get_statevector()
    return statevector.data

def plot_bloch_sphere(statevector, sys_labels):
    """Plota o vetor de estado com labels do sistema físico escolhido."""
    alpha = statevector[0]
    beta = statevector[1]

    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Desenhando a esfera
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='c', alpha=0.15, linewidth=0)

    # Eixos X, Y, Z
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='gray', linestyle='--')
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='gray', linestyle='--')
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='gray', linestyle='--')

    # Vetor de Estado (Flecha Vermelha)
    ax.quiver(0, 0, 0, x, y, z, length=1.0, color='r', arrow_length_ratio=0.1)

    # Usa os labels baseados na escolha do usuário
    ax.text(0, 0, 1.3, sys_labels[0], fontsize=14, ha='center')
    ax.text(0, 0, -1.4, sys_labels[1], fontsize=14, ha='center')
    
    ax.set_title(f'Esfera de Bloch ({sys_labels[2]})', fontsize=16)
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    print("=====================================================")
    print("     Bem-vindo ao Simulador Quântico Interativo!     ")
    print("=====================================================")
    
    print("\n[PASSO 1] Escolha o Sistema Físico para o Qubit:")
    print("  1 - Spin do Elétron (|↑> e |↓>)")
    print("  2 - Polarização do Fóton (|H> e |V>)")
    print("  3 - Nível de Energia Atômico (|g> e |e>)")
    print("  4 - Abstrato Padrão (|0> e |1>)")
    
    sys_choice = input("Digite o número do sistema: ").strip()
    
    labels_dict = {
        '1': ("|↑⟩ (Spin Up)", "|↓⟩ (Spin Down)", "Elétron"),
        '2': ("|H⟩ (Horizontal)", "|V⟩ (Vertical)", "Fóton"),
        '3': ("|g⟩ (Fundamental)", "|e⟩ (Excitado)", "Átomo"),
        '4': ("|0⟩", "|1⟩", "Abstrato")
    }
    sys_labels = labels_dict.get(sys_choice, labels_dict['4'])

    print("\n[PASSO 2] Escolha uma porta (ou combinação) para rotacionar o Qubit:")
    print(" --- Portas Básicas ---")
    print("  'h' : Leva para o eixo +X (Frente)")
    print("  'x' : Inverte o estado (Pólo Sul)")
    print(" --- Combinações para o Equador (Seus círculos verdes) ---")
    print("  '-x' : Aplica X + H para ir ao eixo -X (Fundo)")
    print("  '+y' : Aplica H + S para ir ao eixo +Y (Direita)")
    print("  '-y' : Aplica H + Sdg para ir ao eixo -Y (Esquerda)")

    porta_escolhida = input("\nDigite a sua escolha: ").lower().strip()

    if porta_escolhida not in ['h', 'x', 'y', 'z', '-x', '+y', '-y']:
        print("Opção inválida! Usando a porta 'h' (Hadamard).")
        porta_escolhida = 'h'

    final_state_vector = simulate_qubit_state(gate_name=porta_escolhida)

    prob_0 = np.abs(final_state_vector[0]) ** 2
    prob_1 = np.abs(final_state_vector[1]) ** 2
    
    print("-----------------------------------------------------")
    print("      Probabilidades de Medição      ")
    print(f"--> Chance de medir {sys_labels[0]}: {prob_0:.2%}")
    print(f"--> Chance de medir {sys_labels[1]}: {prob_1:.2%}")
    print("-----------------------------------------------------")

    print("\n--> Gerando visualização 3D...")
    plot_bloch_sphere(final_state_vector, sys_labels)