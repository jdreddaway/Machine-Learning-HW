package experiments.nnet;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.nn.backprop.BackPropagationNetwork;
import shared.DataSet;
import shared.Trainer;


public interface HalterTrainerFactory {

	public Trainer create(BackPropagationNetwork network, DataSet data);
	
	public void serialize(PrintStream writer);
}
