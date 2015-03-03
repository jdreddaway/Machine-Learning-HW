package experiments.boosting;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.DecisionStumpClassifier;
import func.FunctionApproximater;
import func.dtree.GINISplitEvaluator;


public class DecisionStumpClassifierFactory implements FunctionApproximaterFactory {

	@Override
	public FunctionApproximater create() {
		return new DecisionStumpClassifier(new GINISplitEvaluator());
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print("DecisionStumpClassifier");
	}
}
