package optimization_probs;

import java.util.stream.Stream;

import opt.OptimizationAlgorithm;
import shared.Instance;

public class IterationRunner extends AlgorithmRunner {
	
	private OptimizationAlgorithm algo;
	private int numIterations;

	public IterationRunner(String algoName, OptimizationAlgorithm algo, int numIterations) {
		super(algoName);
		this.algo = algo;
		this.numIterations = numIterations;
	}

	/**
	 * @return the optimal instance
	 */
	@Override
	public Stream<Instance> get() {
		for (int i = 0; i < numIterations; i++) {
			algo.train();
		}
		
		return Stream.of(algo.getOptimal());
	}
}
