package experiments.nnet;
import shared.DataSet;
import func.nn.NetworkTrainer;
import func.nn.backprop.BackPropagationNetwork;

public abstract class BackPropagationTrainerFactory {
	
	private String name;

	public BackPropagationTrainerFactory(String name) {
		this.name = name;
	}

	public abstract NetworkTrainer create(DataSet data, BackPropagationNetwork network);

	public String getName() {
		return name;
	}
}
