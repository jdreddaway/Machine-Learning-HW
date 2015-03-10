package optimization_probs;

import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import shared.Instance;

public interface AlgorithmRunnerFactory extends Function<AllTypesOptimizationProblemSupplier, AlgorithmRunner> {
}
