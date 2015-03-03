package experiments.nnet;
import shared.DataSet;
import shared.GradientErrorMeasure;
import shared.SumOfSquaresError;
import func.nn.NetworkTrainer;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.WeightUpdateRule;


public class BatchNetworkTrainerFactory extends BackPropagationTrainerFactory {

	private WeightUpdateRule updateRule;

	public BatchNetworkTrainerFactory(WeightUpdateRule updateRule) {
		super("batch");
		this.updateRule = updateRule;
	}

	@Override
	public NetworkTrainer create(DataSet data, BackPropagationNetwork network) {
		GradientErrorMeasure errorMeasure = new SumOfSquaresError();
		return new BatchBackPropagationTrainer(data, network, errorMeasure, updateRule);
	}

}
