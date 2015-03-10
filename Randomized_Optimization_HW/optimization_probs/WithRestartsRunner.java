package optimization_probs;

import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import opt.OptimizationAlgorithm;
import shared.Instance;

public class WithRestartsRunner extends AlgorithmRunner {
	
	private final Function<AllTypesOptimizationProblemSupplier, OptimizationAlgorithm> algoFactory;
	private final int numRestarts;
	private final int numIterations;
	private final AllTypesOptimizationProblemSupplier problemSupplier;

	public WithRestartsRunner(
			String algoName,
			AllTypesOptimizationProblemSupplier problemSupplier,
			Function<AllTypesOptimizationProblemSupplier, OptimizationAlgorithm> algoFactory,
			int numRestarts,
			int numIterations) {
		super(algoName);
		this.problemSupplier = problemSupplier;
		this.algoFactory = algoFactory;
		this.numRestarts = numRestarts;
		this.numIterations = numIterations;
	}

	@Override
	public Stream<Instance> get() {
		return IntStream.range(0, numRestarts).parallel()
				.mapToObj(i -> runOnce())
				.sequential()
				.reduce(Stream.empty(), Stream::concat);
	}
	
	private Stream<Instance> runOnce() {
		OptimizationAlgorithm algorithm = algoFactory.apply(problemSupplier);
		IterationRunner iterRunner = new IterationRunner(algoName, algorithm, numIterations);
		return iterRunner.get();
	}
}
