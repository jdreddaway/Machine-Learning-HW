package optimization_probs.runners;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import opt.OptimizationAlgorithm;
import shared.Instance;

public class TimedRunner extends AlgorithmRunner {
	
	private final long timeToRun;
	private final OptimizationAlgorithm algo;

	public TimedRunner(OptimizationAlgorithm algo, long timeToRun) {
		super(algo.getClass().getSimpleName());
		this.algo = algo;
		this.timeToRun = timeToRun;
	}

	@Override
	public RunResult get() {
		long start = System.currentTimeMillis();
		
		List<Double> errorPerIteration = new ArrayList<>();
		while (System.currentTimeMillis() - start < timeToRun) {
			errorPerIteration.add(algo.train());
		}

		//TODO account for different length error arrays in RunResult.combine...
		double[] errorPerIterationPrimative = errorPerIteration.stream().mapToDouble(Double::doubleValue).toArray();
		RunResult result = new RunResult(new Instance[] { algo.getOptimal() }, errorPerIterationPrimative);
		return result;
	}

}
