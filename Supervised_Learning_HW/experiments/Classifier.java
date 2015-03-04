package experiments;
import java.io.PrintStream;
import shared.DataSet;


public interface Classifier {

	/**
	 * Should train the classifier as if it were brand new.
	 * @param data
	 */
	public void trainUsing(DataSet data);
	
	public boolean[] evaluate(DataSet data);

	public void serialize(PrintStream writer);
}
