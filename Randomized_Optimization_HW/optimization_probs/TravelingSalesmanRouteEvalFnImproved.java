package optimization_probs;

import opt.example.TravelingSalesmanRouteEvaluationFunction;
import shared.Instance;

public class TravelingSalesmanRouteEvalFnImproved extends TravelingSalesmanRouteEvaluationFunction {

	public TravelingSalesmanRouteEvalFnImproved(double[][] points) {
		super(points);
	}

	@Override
	public double value(Instance d) {
		return -1 / super.value(d);
	}
}
