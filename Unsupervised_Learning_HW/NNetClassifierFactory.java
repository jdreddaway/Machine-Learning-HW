import experiments.Evaluator;
import experiments.nnet.HalterTrainerFactory;
import experiments.nnet.NeuralNetworkClassifierWrapper;
import func.nn.activation.DifferentiableActivationFunction;

public class NNetClassifierFactory {
	
	private final int numHiddenNodes;
	private final DifferentiableActivationFunction activationFn;
	private final HalterTrainerFactory trainerFactory;
	private final Evaluator evaluator;

	public NNetClassifierFactory(int numHiddenNodes, DifferentiableActivationFunction activationFn, HalterTrainerFactory trainerFactory) {
		this.numHiddenNodes = numHiddenNodes;
		this.activationFn = activationFn;
		this.trainerFactory = trainerFactory;
		evaluator = new Evaluator();
	}

	public NeuralNetworkClassifierWrapper create(int numFeatures) {
		return new NeuralNetworkClassifierWrapper(
				evaluator, numHiddenNodes, numFeatures, activationFn, trainerFactory);
	}
}
