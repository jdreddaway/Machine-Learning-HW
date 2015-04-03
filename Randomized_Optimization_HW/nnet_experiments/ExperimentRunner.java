package nnet_experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.function.Function;

import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;

public class ExperimentRunner {
	
	private final BackPropagationNetworkFactory networkFactory;
	private final int numHiddenNodes;
	private final int numIterations;
	private final int numRestarts;
	private final DataSet trainingData;
	private final DataSet testingData;

	public ExperimentRunner(int numHiddenNodes, int numIterations, int numRestarts,
			DataSet trainingData, DataSet testingData) {
		this.numHiddenNodes = numHiddenNodes;
		this.numIterations = numIterations;
		this.numRestarts = numRestarts;
		this.trainingData = trainingData;
		this.testingData = testingData;

		networkFactory = new BackPropagationNetworkFactory();
	}

	private Experiment createExperiment(PrintStream outputStream) {
		return new Experiment(
				outputStream, networkFactory,
				numHiddenNodes, numIterations, numRestarts, trainingData, testingData);
	}
	
	public void runWithRestarts(String outputFilepath,
			AlgorithmFactory<OptimizationAlgorithm> algoFactory) throws FileNotFoundException {
		try (PrintStream outputStream = new PrintStream(new File(outputFilepath))) { 
			Experiment hillclimbingExp = createExperiment(outputStream);
			hillclimbingExp.runWithRestarts(algoFactory);
		}
	}
	
	public void runGenetic(String outputFilepath,
			AlgorithmFactory<StandardGeneticAlgorithmImproved> algoFactory) throws FileNotFoundException {
		try (PrintStream outputStream = new PrintStream(new File(outputFilepath))) { 
			Experiment hillclimbingExp = createExperiment(outputStream);
			hillclimbingExp.runGenetic(algoFactory);
		}
	}
}
