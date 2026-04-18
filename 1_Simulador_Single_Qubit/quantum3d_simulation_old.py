import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Linha de segurança para compatibilidade de gráficos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def simulate_qubit_state(gate_name='h'):
    """
    Simula o estado de um único qubit após a aplicação de uma porta quântica.

    Args:
        gate_name (str): O nome da porta quântica a ser aplicada.
                         Opções válidas: 'h' (Hadamard), 'x' (Pauli-X),
                         'y' (Pauli-Y), 'z' (Pauli-Z).
                         Padrão é 'h'.

    Returns:
        numpy.ndarray: Um array numpy representando o vetor de estado final do qubit.
                       O vetor contém duas amplitudes complexas [alpha, beta],
                       correspondendo aos estados base |0> e |1>.
    """
    qc = QuantumCircuit(1)

    if gate_name == 'h':
        qc.h(0)
    elif gate_name == 'x':
        qc.x(0)
    elif gate_name == 'y':
        qc.y(0)
    elif gate_name == 'z':
        qc.z(0)

    print("--> Circuito Quântico Criado:")
    print(qc)

    qc.save_statevector()

    simulator = AerSimulator()
    result = simulator.run(qc).result()
    statevector = result.get_statevector()
    return statevector.data


def plot_bloch_sphere(statevector):
    """
    Plota o vetor de estado de um qubit em uma Esfera de Bloch 3D.

    Esta função calcula as coordenadas esféricas (theta, phi) a partir
    das amplitudes de probabilidade do vetor de estado e visualiza
    o estado do qubit como um vetor dentro de uma esfera unitária.

    Args:
        statevector (numpy.ndarray): Um vetor de estado de 2 elementos [alpha, beta]
                                     representando o estado quântico.

    Returns:
        None: A função exibe um gráfico interativo usando matplotlib e não retorna valor.
    """
    alpha = statevector[0]
    beta = statevector[1]

    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='c', alpha=0.15, linewidth=0)

    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='gray', linestyle='--')
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='gray', linestyle='--')
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='gray', linestyle='--')

    ax.quiver(0, 0, 0, x, y, z, length=1.0, color='r', arrow_length_ratio=0.1)

    ax.text(0, 0, 1.3, "|0⟩", fontsize=14)
    ax.text(0, 0, -1.4, "|1⟩", fontsize=14)
    ax.set_title('Visualização do Qubit na Esfera de Bloch', fontsize=16)
    ax.set_aspect('equal')
    plt.show()


# --- TERMINAL PRINCIPAL ---
if __name__ == "__main__":
    print("=====================================================")
    print("     Bem-vindo ao Simulador Quântico Interativo!     ")
    print("=====================================================")
    print("Este programa simula um 'qubit', a unidade básica da computação quântica.")
    print("Ele começa no estado |0> (apontando para cima).")
    print("Você pode aplicar uma 'porta quântica' para rotacionar seu estado.")
    print("\nAo rodar este código, você verá que cada porta é uma operação de Álgebra Linear!")
    print("-----------------------------------------------------")

    # Oferece as opções para o usuário
    print("\nEscolha uma porta quântica para aplicar:")
    print("  'h' (Hadamard): Cria uma superposição (50% |0> e 50% |1>).")
    print("  'x' (Pauli-X): Inverte o estado (Também chamado de Inversor).")
    print("  'y' (Pauli-Y): Rotaciona em torno do eixo Y.")
    print("  'z' (Pauli-Z): Aplica uma mudança de fase.")

    porta_escolhida = input("\nDigite a sua escolha (h/x/y/z): ").lower().strip()

    if porta_escolhida not in ['h', 'x', 'y', 'z']:
        print("\nOpção inválida! Usando a porta 'h' (Hadamard) como padrão.")
        porta_escolhida = 'h'

    print(f"\n--> Iniciando a simulação com a porta '{porta_escolhida}'...")

    print(f"\n--> Iniciando a simulação com a porta '{porta_escolhida}'...")
    final_state_vector = simulate_qubit_state(gate_name=porta_escolhida)

    print(f"--> Vetor de Estado Final: {final_state_vector}")

    # --- A NOVA FUNCIONALIDADE ---
    # O vetor de estado é [alpha, beta]. A probabilidade é o módulo ao quadrado de cada componente.
    prob_0 = np.abs(final_state_vector[0]) ** 2
    prob_1 = np.abs(final_state_vector[1]) ** 2
    print("-----------------------------------------------------")
    print("      Probabilidades de Medição      ")
    print(f"--> Chance de encontrar o qubit no estado |0>: {prob_0:.2%}")
    print(f"--> Chance de encontrar o qubit no estado |1>: {prob_1:.2%}")
    print("-----------------------------------------------------")
    # ----------------------------------------

    print("\n--> Gerando visualização 3D...")
    plot_bloch_sphere(final_state_vector)

    print("--> Simulação concluída.")

