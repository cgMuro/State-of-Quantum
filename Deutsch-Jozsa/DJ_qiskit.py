# Deutsch-Jozsa algorithm implementation with Qiskit
# Main reference used: https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html

import numpy as np
import qiskit
from qiskit.circuit.gate import Gate
from qiskit.providers.ibmq import IBMQ, least_busy
from qiskit.visualization import plot_histogram


class DeutschJozsa():
    """ Implements the Deutsch-Jozsa algorithm logic. """
    def __init__(
        self, 
        type: str,      # Type of oracle to implement: either CONSTANT or BALANCED
        num_bits: int   # Number of bits for the input string
    ) -> None:
        
        # Check if the right type of oracle was passed
        assert type == 'constant' or type == 'balanced', f'type should be either constant or balanced. {type} not allowed'

        self.type = type
        self.num_bits  = num_bits

    def oracle(self) -> Gate:
        """ Based on the type of oracle creates a quantum oracle with n input qubits and 1 output qubit. """
        # Create circuit
        circuit = qiskit.QuantumCircuit(self.num_bits+1)

        # CONSTANT ORACLE
        if self.type == 'constant':
            # Randomly set the output qubit to be 0 or 1
            output = np.random.randint(2)
            if output == 1:
                circuit.x(self.num_bits)

        # BALANCED ORACLE
        # To create a balanced oracle we need to perform CNOTs with each input qubit as a control and the output bit as the target.
        # To vary the input state we wrap some of the controls in X-gates.
        if self.type == 'balanced':
            # Randomly generate a binary string -> decides which controls to wrap
            b = np.random.randint(1, 2**self.num_bits)
            binary_string = format(b, '0' + str(self.num_bits) + 'b')

            # Place X-gates
            for qubit in range(len(binary_string)):
                if binary_string[qubit] == '1':
                    circuit.x(qubit)

            # Controlled-NOT gates
            for qubit in range(self.num_bits):
                circuit.cx(qubit, self.num_bits)

            # Place X-gates
            for qubit in range(len(binary_string)):
                if binary_string[qubit] == '1':
                    circuit.x(qubit)

        # Create gate and name it
        oracle_gate = circuit.to_gate()
        oracle_gate.name = 'Oracle'

        return oracle_gate

    def algorithm(self, oracle) -> qiskit.QuantumCircuit:
        """ Performs the Deutsch-Jozsa algorithm with the given oracle. """
        # Create circuit
        circuit = qiskit.QuantumCircuit(
            qiskit.QuantumRegister(self.num_bits+1), 
            qiskit.ClassicalRegister(self.num_bits)
        )

        # Put input qubits in state |+⟩, i.e. apply H-gates
        for qubit in range(self.num_bits):
            circuit.h(qubit)

        # Put output qubit in state |−⟩
        circuit.x(self.num_bits)
        circuit.h(self.num_bits)

        # Add oracle
        circuit.append(oracle, range(self.num_bits+1))

        # Repeat H-gates
        for qubit in range(self.num_bits):
            circuit.h(qubit)

        # Measure
        for i in range(self.num_bits):
            circuit.measure(i, i)

        return circuit

    def simulation(self, circuit, plot: bool):
        """ Performs simulation on the given circuit. """
        # Get simulator and serialize qobj
        qasm_simulator = qiskit.Aer.get_backend('qasm_simulator')
        transpiled_circuit = qiskit.compiler.transpiler.transpile(circuit, qasm_simulator)
        qobj = qiskit.compiler.assembler.assemble(transpiled_circuit)

        # Get results
        results = qasm_simulator.run(qobj).result()
        answer = results.get_counts()

        # Plot histogram if requested
        if plot:
            plot_histogram(answer)

        return answer

    def run_quantum_hardware(self, circuit, plot: bool):
        """ Run the given circuit on the IBM quantum hardware in the cloud. """
        # Load IBMQ account and get the least busy backend device with greater than or equal to (n+1) qubits
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (self.num_bits+1) and not x.configuration().simulator and x.status().operational == True))
        print('Least busy backend:', backend)

        # Run our circuit on the least busy backend and monitor the execution of the job in the queue
        transpiled_circuit = qiskit.compiler.transpiler.transpile(circuit, backend, optimization_level=3)
        qobj = qiskit.compiler.assembler.assemble(transpiled_circuit, backend)
        job = backend.run(qobj)
        qiskit.tools.monitor.job_monitor(job, interval=2)

        # Get results
        results = job.result()
        answer = results.get_counts()

        # Plot histogram if requested
        if plot:
            plot_histogram(answer)

        return answer
