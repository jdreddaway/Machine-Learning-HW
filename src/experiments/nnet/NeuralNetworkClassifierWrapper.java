package experiments.nnet;
import java.io.PrintStream;
import java.io.PrintWriter;

import shared.DataSet;
import shared.Instance;
import shared.Trainer;
import supervised_experiments.Classifier;
import supervised_experiments.Evaluator;
import dist.DiscreteDistribution;
import func.nn.Link;
import func.nn.activation.DifferentiableActivationFunction;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;


public class NeuralNetworkClassifierWrapper implements Classifier {
	
	private BackPropagationNetwork network;

	private final Evaluator evaluator;
	private final HalterTrainerFactory trainerFactory;
	private final int hiddenNodeCount;
	private final int numInputs;
	private final DifferentiableActivationFunction activationFunction;

	public NeuralNetworkClassifierWrapper(Evaluator evaluator, int hiddenNodeCount,
			int numInputs, DifferentiableActivationFunction activationFunction, HalterTrainerFactory trainerFactory) {
		this.evaluator = evaluator;
		this.hiddenNodeCount = hiddenNodeCount;
		this.numInputs = numInputs;
		this.activationFunction = activationFunction;
		this.trainerFactory = trainerFactory;
	}

	@Override
	public void trainUsing(DataSet data) {
		//System.out.println("Network before: " + getNetworkAsString());
		createNetwork();
		Trainer trainer = trainerFactory.create(network, data);
		double error = trainer.train();
		System.out.println("Total error corrected: " + error);
	}

	@Override
	public boolean[] evaluate(DataSet data) {
		//System.out.println("Network after: " + getNetworkAsString());
		return evaluator.evaluate(data, this::value);
	}
	
	private Instance value(Instance input) {
		network.setInputValues(input.getData());
        network.run();
        boolean output = network.getBinaryOutputValue();
		return new Instance(output);
	}
	
	public String getNetworkAsString() {
		StringBuilder strBuilder = new StringBuilder();
		for (Object oLink : network.getLinks()) {
			Link link = (Link) oLink;
			
			strBuilder.append(link.getWeight());
			strBuilder.append(',');
		}
		
		return strBuilder.toString();
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print(activationFunction.getClass().getCanonicalName() + ",");
		writer.print(hiddenNodeCount + ",");
		trainerFactory.serialize(writer);
		//TODO consider viewing contribution of each input (need to calculate contribution of each node)
	}
	
	private void createNetwork() {
		int[] topology;
        if (hiddenNodeCount != 0) {
            topology = new int[3];
            topology[1] = hiddenNodeCount;
        } else {
            topology = new int[2];
        }
        topology[0] = numInputs;
		topology[topology.length - 1] = 1;

        network =  (new BackPropagationNetworkFactory())
            .createClassificationNetwork(topology, activationFunction);
	}

	/*
	 * WHAT TO REPORT:
	 * 
	 * graph % correctly labeled vs. # of epochs vs. learning vs. momentum
	 * 
	 * graph training data accuracy and testing data accuracy vs. # of epochs
	 */
}
