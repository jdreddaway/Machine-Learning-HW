package experiments;
import java.io.PrintStream;
import java.io.PrintWriter;


public class Results {

	private boolean[] trainingResults;
	private boolean[] testResults;
	private long trainingTime;
	private long testTime;

	public Results(long trainingTime, long testTime, boolean[] trainingResults, boolean[] testResults) {
		this.trainingResults = trainingResults;
		this.testResults = testResults;
		this.trainingTime = trainingTime;
		this.testTime = testTime;
	}
	
	public void print(PrintStream writer) {
		String accuracyFormat = "%1.4f";
		writer.printf(accuracyFormat, calcAccuracy(trainingResults));
		writer.print(',');
		writer.printf(accuracyFormat, calcAccuracy(testResults));
		writer.print(',');
		writer.print(trainingTime);
		writer.print(',');
		writer.print(testTime);
		writer.print(',');
		printResults(writer, trainingResults);
		writer.print(',');
		printResults(writer, testResults);
	}
	
	public void printLabels(PrintWriter writer) {
		writer.print("training accuracy,test accuracy,training time,test time,training results,test results");
	}
	
	private double calcAccuracy(boolean[] results) {
		int numCorrect = 0;
		
		for (boolean isCorrect : results) {
			if (isCorrect) {
				numCorrect++;
			}
		}
		
		return 1.0 * numCorrect / results.length;
	}
	
	private void printResults(PrintStream writer, boolean[] results) {
		for (boolean isCorrect : results) {
			writer.print(isCorrect ? '1' : '0');
		}
	}
}
