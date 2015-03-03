package experiments.knn;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.KNNClassifier;
import shared.DataSet;
import shared.DistanceMeasure;
import supervised_experiments.Classifier;
import supervised_experiments.Evaluator;


public class KnnClassifierWrapper implements Classifier {
	
	private KNNClassifier classifier;

	private final int k;
	private final boolean weight;
	private final DistanceMeasure measure;
	private final double range;
	private final Evaluator evaluator;
	
	public KnnClassifierWrapper(Evaluator evaluator, int k, boolean weight, DistanceMeasure measure, double range) {
		this.evaluator = evaluator;
		this.k = k;
		this.weight = weight;
		this.measure = measure;
		this.range = range;
	}

	@Override
	public void trainUsing(DataSet data) {
		classifier = new KNNClassifier(k, weight, measure, range);
		classifier.estimate(data);
	}

	@Override
	public boolean[] evaluate(DataSet data) {
		return evaluator.evaluate(data, classifier::value);
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print(k);
		writer.print(',');
		writer.print(weight);
		writer.print(',');
		writer.print(measure.getClass().getCanonicalName());
		writer.print(',');
		writer.print(range);
	}

	/*
	 * WHAT TO REPORT:
	 * % correct for test & train vs. # of nearest neighbors
	 * weighted vs. unweighted
	 */
}
