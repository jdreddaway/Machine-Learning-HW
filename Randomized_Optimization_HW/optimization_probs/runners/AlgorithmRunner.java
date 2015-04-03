package optimization_probs.runners;

import java.util.function.Supplier;
import java.util.stream.Stream;

import shared.Instance;

public abstract class AlgorithmRunner implements Supplier<RunResult>{

	public final String algoName;

	public AlgorithmRunner(String algoName) {
		this.algoName = algoName;
	}
}
