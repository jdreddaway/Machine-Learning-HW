package optimization_probs.runners;

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
	public RunResult get() {
		double[] errorPerIteration = new double[numIterations];

		for (int i = 0; i < numIterations; i++) {
			errorPerIteration[i] = algo.train();
		}
		
		RunResult result = new RunResult(new Instance[] { algo.getOptimal() }, errorPerIteration);
		return result;
	}
}
