package optimization_probs;

import java.util.function.Supplier;
import java.util.stream.Stream;

import shared.Instance;

public abstract class AlgorithmRunner implements Supplier<Stream<Instance>>{

	public final String algoName;

	public AlgorithmRunner(String algoName) {
		this.algoName = algoName;
	}
}
