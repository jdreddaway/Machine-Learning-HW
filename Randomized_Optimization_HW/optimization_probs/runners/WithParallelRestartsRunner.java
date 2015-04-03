package optimization_probs.runners;

import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import opt.OptimizationAlgorithm;
import optimization_probs.AllTypesOptimizationProblemSupplier;

public class WithParallelRestartsRunner extends AlgorithmRunner {
	
	private final Function<AllTypesOptimizationProblemSupplier, OptimizationAlgorithm> algoFactory;
	private final int numRestarts;
	private final AllTypesOptimizationProblemSupplier problemSupplier;
	private final Function<OptimizationAlgorithm, AlgorithmRunner> underlyingRunnerSupplier;

	public WithParallelRestartsRunner(
			String algoName,
			AllTypesOptimizationProblemSupplier problemSupplier,
			Function<AllTypesOptimizationProblemSupplier, OptimizationAlgorithm> algoFactory,
			int numRestarts,
			Function<OptimizationAlgorithm, AlgorithmRunner> underlyingRunnerSupplier) {
		super(algoName);
		this.problemSupplier = problemSupplier;
		this.algoFactory = algoFactory;
		this.numRestarts = numRestarts;
		this.underlyingRunnerSupplier = underlyingRunnerSupplier;
	}

	@Override
	public RunResult get() {
		return IntStream.range(0, numRestarts).parallel()
				.mapToObj(i -> runOnce())
				.sequential()
				.reduce(this::combineResults)
				.get();
	}
	
	private RunResult runOnce() {
		OptimizationAlgorithm algorithm = algoFactory.apply(problemSupplier);
		AlgorithmRunner iterRunner = underlyingRunnerSupplier.apply(algorithm);
		return iterRunner.get();
	}
	
	private RunResult combineResults(RunResult result1, RunResult result2) {
		return result1.combineWithUsingWeightedAverage(result2);
	}
}
