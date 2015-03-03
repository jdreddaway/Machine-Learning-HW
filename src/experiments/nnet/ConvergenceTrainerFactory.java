package experiments.nnet;
import java.io.PrintStream;
import java.io.PrintWriter;

import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Trainer;
import func.nn.backprop.BackPropagationNetwork;


public class ConvergenceTrainerFactory implements HalterTrainerFactory {
	
	private double threshold;
	private int maxIterations;
	private BackPropagationTrainerFactory underlyingTrainerFactory;

	public ConvergenceTrainerFactory(BackPropagationTrainerFactory underlyingTrainerFactory, double threshold, int maxIterations) {
		this.underlyingTrainerFactory = underlyingTrainerFactory;
		this.threshold = threshold;
		this.maxIterations = maxIterations;
	}

	@Override
	public Trainer create(BackPropagationNetwork network, DataSet data) {
		return new ConvergenceTrainer(underlyingTrainerFactory.create(data, network), threshold, maxIterations);
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print("Convergence,");
		writer.print(threshold);
		writer.print(',');
		writer.print(maxIterations);
		writer.print(',');
		writer.print(underlyingTrainerFactory.getName());
	}

}
