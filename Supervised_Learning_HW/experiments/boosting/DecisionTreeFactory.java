package experiments.boosting;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.DecisionTreeClassifier;
import func.FunctionApproximater;
import func.dtree.ChiSquarePruningCriteria;
import func.dtree.SplitEvaluator;

public class DecisionTreeFactory implements FunctionApproximaterFactory {
	
	private SplitEvaluator splitEvaluator;
	private boolean useBinarySplits;
	private int confidence;

	public DecisionTreeFactory(SplitEvaluator splitEvaluator, int confidence, boolean useBinarySplits) {
		this.splitEvaluator = splitEvaluator;
		this.confidence = confidence;
		this.useBinarySplits = useBinarySplits;
	}

	@Override
	public FunctionApproximater create() {
		return new DecisionTreeClassifier(splitEvaluator, new ChiSquarePruningCriteria(confidence), useBinarySplits);
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print("DecisionTreeClassifier,");
		writer.print(confidence);
		writer.print(',');
		writer.print(useBinarySplits);
	}

}
