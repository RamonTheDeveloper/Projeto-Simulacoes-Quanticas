# Importações necessárias para este experimento
import matplotlib
matplotlib.use('TkAgg')
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_qsphere
import matplotlib.pyplot as plt


def create_bell_state():
    """
    Cria um circuito quântico que gera um Estado de Bell,
    o exemplo mais famoso de emaranhamento.
    """

    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.z(1)

    return bell_circuit


if __name__ == "__main__":
    print("--> Iniciando a simulação de emaranhamento quântico...")

    # 1. Cria o circuito do Estado de Bell
    my_circuit = create_bell_state()

    print("\n--> Circuito de Emaranhamento (Estado de Bell) Criado:")
    print(my_circuit)

    # 2. Prepara o simulador para obter o vetor de estado
    simulator = AerSimulator()
    my_circuit.save_statevector()  # Importante para obter o estado
    result = simulator.run(my_circuit).result()
    statevector = result.get_statevector()

    print(f"\n--> Vetor de Estado Final: {statevector}")
    print("\n--> Gerando visualização 3D (Q-Sphere)...")

    # 3. Simulação de Q-Sphere (Sem medição)
    my_circuit = create_bell_state()
    simulator = AerSimulator()
    my_circuit.save_statevector()
    result = simulator.run(my_circuit).result()
    statevector = result.get_statevector()
    fig = plot_state_qsphere(statevector)
    fig.savefig('qsphere.png')
    plt.show()

    # 4. Simulação com medição (Contagens)
    from qiskit import ClassicalRegister

    # Novo circuito com medição
    circuit_with_measure = create_bell_state()
    cr = ClassicalRegister(2)
    circuit_with_measure.add_register(cr)
    circuit_with_measure.measure([0, 1], [0, 1])

    result_counts = simulator.run(circuit_with_measure, shots=1024).result()
    counts = result_counts.get_counts()
    print("\nContagens de medição:", counts)

    print("\n--> Simulação concluída.")
    print("Observe na Q-Sphere: temos dois pontos grandes e opostos (|00> e |11>),")
    print("indicando que o sistema está em uma superposição desses dois estados emaranhados.")