package experiments.boosting;
import java.io.PrintStream;
import java.io.PrintWriter;

import shared.DataSet;
import supervised_experiments.Classifier;
import supervised_experiments.Evaluator;


public class BoostingClassifierWrapper implements Classifier {
	
	private CustomAdaBoostClassifier classifier;

	private final Evaluator evaluator;
	private final FunctionApproximaterFactory funcApproximaterFactory;
	private final int size;

	public BoostingClassifierWrapper(Evaluator evaluator, int size,
			FunctionApproximaterFactory funcApproximaterFactory) {
		this.evaluator = evaluator;
		this.size = size;
		this.funcApproximaterFactory = funcApproximaterFactory;
	}

	@Override
	public void trainUsing(DataSet data) {
		classifier = new CustomAdaBoostClassifier(size, funcApproximaterFactory::create);
		classifier.estimate(data);
	}

	@Override
	public boolean[] evaluate(DataSet data) {
		return evaluator.evaluate(data, classifier::value);
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print(classifier.getSize());
		writer.print(',');
		funcApproximaterFactory.serialize(writer);
	}

	/*
	 * WHAT TO REPORT:
	 * 
	 * prune, dtree pruning confidence, boost iterations, train %, test %
	 */
}
