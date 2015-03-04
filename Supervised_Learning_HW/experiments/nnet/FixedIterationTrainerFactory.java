package experiments.nnet;
import java.io.PrintStream;
import java.io.PrintWriter;

import shared.DataSet;
import shared.FixedIterationTrainer;
import shared.Trainer;
import func.nn.backprop.BackPropagationNetwork;


public class FixedIterationTrainerFactory implements HalterTrainerFactory {
	
	private int numIterations;
	private BackPropagationTrainerFactory underlyingTrainerFactory;

	public FixedIterationTrainerFactory(int numIterations, BackPropagationTrainerFactory underlyingTrainerFactory) {
		this.numIterations = numIterations;
		this.underlyingTrainerFactory = underlyingTrainerFactory;
	}

	@Override
	public Trainer create(BackPropagationNetwork network, DataSet data) {
		Trainer underlyingTrainer = underlyingTrainerFactory.create(data, network);
		return new FixedIterationTrainer(underlyingTrainer, numIterations);
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print(underlyingTrainerFactory.getName() + ",");
		writer.print(numIterations);
	}
}
