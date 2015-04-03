package optimization_probs.runners;

import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import optimization_probs.AllTypesOptimizationProblemSupplier;
import shared.Instance;

public interface AlgorithmRunnerFactory extends Function<AllTypesOptimizationProblemSupplier, AlgorithmRunner> {
}
