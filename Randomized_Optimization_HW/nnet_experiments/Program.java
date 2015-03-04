package nnet_experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import shared.DataSet;
import data.UciDataReader;
import func.nn.backprop.BackPropagationNetworkFactory;

public class Program {
	
	public static void main(String[] args) throws FileNotFoundException {
		String trainingPath = "higgs/training1000.csv";
		String testingPath = "higgs/testing1000.csv";
		int numHiddenNodes = 15;
		int numIterations = 3000;
		int numRestarts = 1000;

		DataSet trainingData = new UciDataReader(new File(trainingPath)).read();
		DataSet testingData = new UciDataReader(new File(testingPath)).read();

		BackPropagationNetworkFactory networkFactory = new BackPropagationNetworkFactory();
		PrintStream outputStream = new PrintStream(new File("output/random_hillclimbing.out")); // output folder needs to already exist
		RandomizedHillclimbingExperiment hillclimbingExp = new RandomizedHillclimbingExperiment(
				outputStream, networkFactory, numHiddenNodes, numIterations, numRestarts, trainingData, testingData);
		hillclimbingExp.runWithRestarts();
	}
	
}
