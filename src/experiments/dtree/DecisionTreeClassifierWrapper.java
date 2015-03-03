package experiments.dtree;
import java.io.PrintStream;
import java.io.PrintWriter;

import shared.DataSet;
import supervised_experiments.Classifier;
import supervised_experiments.Evaluator;
import func.DecisionTreeClassifier;
import func.dtree.ChiSquarePruningCriteria;
import func.dtree.GINISplitEvaluator;
import func.dtree.PruningCriteria;
import func.dtree.SplitEvaluator;


public class DecisionTreeClassifierWrapper implements Classifier {
	private final boolean prune;
	private final int confidence;
	private final Evaluator evaluator;
	private final boolean useBinarySplits;
	private final GINISplitEvaluator splitEvaluator;

	private DecisionTreeClassifier classifier;

	/**
	 * 
	 * @param evaluator
	 * @param useBinarySplits
	 * @param prune
	 * @param confidence
	 */
	public DecisionTreeClassifierWrapper(Evaluator evaluator, boolean useBinarySplits, boolean prune, int confidence) {
		this.evaluator = evaluator;
		this.useBinarySplits = useBinarySplits;
		this.prune = prune;
		this.confidence = confidence;
		splitEvaluator = new GINISplitEvaluator();
	}

	@Override
	public void trainUsing(DataSet data) {
		if (prune) {
			PruningCriteria pruningCriteria = new ChiSquarePruningCriteria(confidence);
			classifier = new DecisionTreeClassifier(splitEvaluator, pruningCriteria, useBinarySplits);
		} else {
			classifier = new DecisionTreeClassifier(splitEvaluator, useBinarySplits);
		}
		classifier.estimate(data);
	}

	@Override
	public boolean[] evaluate(DataSet data) {
		return evaluator.evaluate(data, (i) -> classifier.value(i));
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print(classifier.isUseBinarySplits());
		writer.print(',');
		writer.print(prune);
		writer.print(',');
		writer.print(confidence);
		writer.print(',');
		writer.print(classifier.getHeight());
		writer.print(',');
		writer.print(classifier.getRoot().getSplitStatistics().getBranchCount());
	}
}
