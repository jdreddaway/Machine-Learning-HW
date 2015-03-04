package experiments.nnet;
import shared.DataSet;
import shared.GradientErrorMeasure;
import shared.SumOfSquaresError;
import func.nn.NetworkTrainer;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.StochasticBackPropagationTrainer;
import func.nn.backprop.WeightUpdateRule;


public class StochasticNetworkTrainerFactory extends BackPropagationTrainerFactory {
	private WeightUpdateRule updateRule;

	public StochasticNetworkTrainerFactory(WeightUpdateRule updateRule) {
		super("stochastic");
		this.updateRule = updateRule;
	}

	@Override
	public NetworkTrainer create(DataSet data, BackPropagationNetwork network) {
		GradientErrorMeasure errorMeasure = new SumOfSquaresError();
		return new StochasticBackPropagationTrainer(data, network, errorMeasure, updateRule);
	}

}
