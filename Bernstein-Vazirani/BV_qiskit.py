# Bernstein-Vazirani algorithm implementation with Qiskit
# Main resource used: https://qiskit.org/textbook/ch-algorithms/bernstein-vazirani.html

import qiskit


class BernsteinVazirani():
    """ Implements the Bernstein-Vazirani algorithm logic. """
    def __init__(
        self,
        num_qubits: int,    # Number of qubits in the circuit
        bitstring: str  # Input bits string
    ) -> None:
        
        self.n = num_qubits
        self.bitstring = bitstring

    def algorithm(self, draw: bool = False) -> qiskit.QuantumCircuit:
        """ Returns the circuit for performing the Bernstein-Vazirani algorithm. """
        # Create a circuit with: 
        #       - n input qubits + 1 output qubit
        #       - n classical bits for storing the measurement output
        circuit = qiskit.QuantumCircuit(
            qiskit.QuantumRegister(self.n+1), 
            qiskit.ClassicalRegister(self.n)
        )

        # Put output qubit in state |−⟩
        circuit.h(self.n)
        circuit.z(self.n)

        # Apply Hadamard gates
        for qubit in range(self.n):
            circuit.h(qubit)

        # Apply visual barrier 
        circuit.barrier()

        # Apply the inner product oracle
        s = self.bitstring[::-1]  # Reverse bitstring to fit qiskit's qubit ordering
        for qubit in range(self.n):
            if s[qubit] == '0':
                circuit.i(qubit)
            else:
                circuit.cx(qubit, self.n)

        # Apply visual barrier 
        circuit.barrier()
            
        # Apply Hadamard gates
        for qubit in range(self.n):
            circuit.h(qubit)

        # Measurement
        for qubit in range(self.n):
            circuit.measure(qubit, qubit)

        # Print and draw circuit if requested
        if draw:
            print(circuit)
            circuit.draw()

        return circuit

    def simulation(self, circuit: qiskit.QuantumCircuit, plot: bool = False):
        """ Performs simulation on the given circuit. """
        # Get simulator and serialize qobj
        qasm_sim = qiskit.Aer.get_backend('qasm_simulator')
        qobj = qiskit.assemble(circuit)
        # Get results
        results = qasm_sim.run(qobj).result()
        answer = results.get_counts()

        # Plot histogram if requested
        if plot:
            qiskit.visualization.plot_histogram(answer)

        return answer

    def run_quantum_hardware(self, circuit: qiskit.QuantumCircuit, plot: bool = False):
        """ Run the given circuit on the IBM quantum hardware in the cloud. """
        # Load IBMQ account and get the least busy backend device with greater than or equal to (n+1) qubits
        qiskit.providers.ibmq.IBMQ.load_account()
        provider = qiskit.providers.ibmq.IBMQ.get_provider(hub='ibm-q')
        backend = qiskit.providers.ibmq.least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (self.num_bits+1) and not x.configuration().simulator and x.status().operational == True))
        print('Least busy backend:', backend)

        # Run circuit on the least busy backend and monitor the execution of the job in the queue
        transpiled_circuit = qiskit.compiler.transpiler.transpile(circuit, backend, optimization_level=3)
        qobj = qiskit.compiler.assembler.assemble(transpiled_circuit, backend)
        job = backend.run(qobj)
        qiskit.tools.monitor.job_monitor(job, interval=2)

        # Get results
        results = job.result()
        answer = results.get_counts()

        # Plot histogram if requested
        if plot:
            qiskit.visualization.plot_histogram(answer)

        return answer
