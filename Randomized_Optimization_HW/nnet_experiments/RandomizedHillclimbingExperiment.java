package nnet_experiments;

import java.io.PrintStream;
import java.util.stream.IntStream;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.DataSetDescription;
import shared.SumOfSquaresError;
import func.nn.NeuralNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;

public class RandomizedHillclimbingExperiment {
	
	private final PrintStream outputStream;
	private final BackPropagationNetworkFactory networkFactory;
	private final int numHiddenNodes;
	private final int numIterations;
	private final DataSet trainingData;
	private final DataSet testingData;
	private final int numRestarts;

	public RandomizedHillclimbingExperiment(PrintStream outputStream, BackPropagationNetworkFactory networkFactory,
			int numHiddenNodes, int numIterations, int numRestarts, DataSet trainingData, DataSet testingData) {
		this.outputStream = outputStream;
		this.networkFactory = networkFactory;
		this.numHiddenNodes = numHiddenNodes;
		this.numIterations = numIterations;
		this.numRestarts = numRestarts;
		this.trainingData = trainingData;
		this.testingData = testingData;
	}
	
	public void runWithRestarts() {
		long start = System.currentTimeMillis();
		NetworkPerformance best = IntStream.range(0, numRestarts).parallel()
				.mapToObj(i -> run())
				.max(NetworkPerformance::compareByTestingPerf).get();
		long end = System.currentTimeMillis();
		
		System.out.println("Best Training: " + best.numTrainingCorrect);
		System.out.println("Best Testing: " + best.numTestingCorrect);
		
		System.out.println("Time: " + (end - start) / 1000 + " seconds");
	}

	private NetworkPerformance run() {
		NeuralNetwork network = createNetwork();
		NeuralNetworkOptimizationProblem problem = new NeuralNetworkOptimizationProblem(trainingData, network, new SumOfSquaresError());
		RandomizedHillClimbing algorithm = new RandomizedHillClimbing(problem);
		
		runAlgorithm(algorithm);
		int numTrainingCorrect = countNumCorrect(network, trainingData);
		int numTestingCorrect = countNumCorrect(network, testingData);
		
		printResult(numTrainingCorrect, numTestingCorrect);
		
		System.out.println("Training Correct: " + numTrainingCorrect);
		System.out.println("Testing Correct: " + numTestingCorrect);
		System.out.println();
		
		return new NetworkPerformance(numTrainingCorrect, numTestingCorrect);
	}
	
	private synchronized void printResult(int numTrainingCorrect, int numTestingCorrect) {
		outputStream.println(numTrainingCorrect + "," + numTestingCorrect);
	}
	
	private static int countNumCorrect(NeuralNetwork network, DataSet data) {
		int numCorrect = 0;
		for(int i = 0; i < data.size(); i++) {
            network.setInputValues(data.get(i).getData());
            network.run();

            boolean actualCategory = data.get(i).getLabel().getBoolean();
            boolean outputCategory = network.getOutputValues().get(0) > 0.5;
            
            if (actualCategory == outputCategory) {
            	numCorrect++;
            }
        }

		return numCorrect;
	}
	
	private void runAlgorithm(OptimizationAlgorithm algo) {
		for(int i = 0; i < numIterations; i++) {
            algo.train();
		}
	}

	private NeuralNetwork createNetwork() {
		int numAttributes = new DataSetDescription(trainingData).getAttributeCount();
		int[] layerNodeCounts = { numAttributes, numHiddenNodes, 1 };
		NeuralNetwork network = networkFactory.createClassificationNetwork(layerNodeCounts);

		return network;
	}
}
