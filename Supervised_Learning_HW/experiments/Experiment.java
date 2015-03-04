package experiments;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.Collection;

import data.UciDataReader;
import shared.DataSet;


public class Experiment {

	private String[] trainingFilepaths;
	private String testFilepath;

	public Experiment(String[] trainingFilepaths, String testFilepath) {
		this.trainingFilepaths = trainingFilepaths;
		this.testFilepath = testFilepath;
	}
	
	public void run(PrintStream outputStream, Classifier classifier) throws FileNotFoundException {
		for (String trainingFilepath : trainingFilepaths) {
			run(outputStream, trainingFilepath, classifier);
		}
	}
	
	public void runAll(Collection<? extends Classifier> classifiers, String outputFilename) throws FileNotFoundException {
		try(PrintStream resultsWriter = new PrintStream(outputFilename)) {
			int currentClassifierNum = 1;
			for (Classifier classifier : classifiers) {
				System.out.println("Executing classifier " + currentClassifierNum + " of " + classifiers.size());
				currentClassifierNum++;
				
				classifier.serialize(System.out);
				System.out.println();
				
				long start = System.currentTimeMillis();
				run(resultsWriter, classifier);
				long end = System.currentTimeMillis();
				long time = (end - start) / 1000;

				System.out.println("Took " + time + " seconds.\n");
			}
		}
	}
	
	private void run(PrintStream outputWriter, String trainingFilepath, Classifier classifier) throws FileNotFoundException {
		System.out.println(trainingFilepath);
		
		UciDataReader trainingDataReader = new UciDataReader(new File(trainingFilepath));
		UciDataReader testDataReader = new UciDataReader(new File(testFilepath));
		DataSet trainingData = trainingDataReader.read();

		long start = System.currentTimeMillis();
		classifier.trainUsing(trainingData);
		long end = System.currentTimeMillis();
		long trainingTime = end - start;
		
		boolean[] trainingResults = classifier.evaluate(trainingData);

		DataSet testData = testDataReader.read();
		start = System.currentTimeMillis();
		boolean[] testResults = classifier.evaluate(testData);
		end = System.currentTimeMillis();
		long testTime = end - start;
		
		Results results = new Results(trainingTime, testTime, trainingResults, testResults);
		printResults(outputWriter, trainingFilepath, classifier, results);
	}

	private void printResults(PrintStream writer, String trainingFilepath, Classifier classifier, Results results) {
		writer.print(trainingFilepath + ",");
		classifier.serialize(writer);
		writer.print(',');
		results.print(writer);
		writer.println();
	}
}
