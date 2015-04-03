package optimization_probs.runners;

import java.io.PrintStream;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import opt.EvaluationFunction;
import shared.Instance;

public class RunResult {

	private final Instance[] finalInstances;
	private final double[] errorPerIteration;

	public RunResult(Instance[] finalInstances, double[] errorPerIteration) {
		this.finalInstances = finalInstances;
		this.errorPerIteration = errorPerIteration;
	}
	
	public RunResult combineWithUsingWeightedAverage(RunResult other) {
		Instance[] instances = concatInstances(other.finalInstances);
		double[] errors = combineErrorsUsingWeightedAverage(other);
		return new RunResult(instances, errors);
	}
	
	public DoubleStream evaluateInstances(EvaluationFunction evalFn) {
		return Stream.of(finalInstances).parallel().mapToDouble(evalFn::value);
	}
	
	private double[] combineErrorsUsingWeightedAverage(RunResult other) {
		double[] longestErrors;
		int shortestLength;
		
		if (errorPerIteration.length < other.errorPerIteration.length) {
			shortestLength = errorPerIteration.length;
			longestErrors = other.errorPerIteration;
		} else {
			shortestLength = other.errorPerIteration.length;
			longestErrors = errorPerIteration;
		}

		double[] combined = new double[longestErrors.length];
		int totalNumInstances = finalInstances.length + other.finalInstances.length;
		double myWeight = 1.0 * finalInstances.length / totalNumInstances;
		double otherWeight = 1.0 * other.finalInstances.length / totalNumInstances;

		int i = 0;
		for (; i < shortestLength; i++) {
			combined[i] = errorPerIteration[i] * myWeight + other.errorPerIteration[i] * otherWeight;
		}
		
		for (; i < longestErrors.length; i++) {
			combined[i] = longestErrors[i];
		}
		
		return combined;
	}
	
	private Instance[] concatInstances(Instance[] other) {
	   int len1 = finalInstances.length;
	   int len2 = other.length;
	   Instance[] combined = new Instance[len1 + len2];
	   System.arraycopy(finalInstances, 0, combined, 0, len1);
	   System.arraycopy(other, 0, combined, len1, len2);
	   return combined;
	}

	public void printErrors(PrintStream errorWriter) {
		for (double each : errorPerIteration) {
			errorWriter.println(each);
		}
	}
}
