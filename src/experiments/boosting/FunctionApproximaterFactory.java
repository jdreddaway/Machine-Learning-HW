package experiments.boosting;
import java.io.PrintStream;
import func.FunctionApproximater;


public interface FunctionApproximaterFactory {
	
	public FunctionApproximater create();

	public void serialize(PrintStream writer);
}
