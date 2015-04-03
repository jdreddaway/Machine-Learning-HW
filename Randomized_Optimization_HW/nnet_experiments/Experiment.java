package nnet_experiments;

import java.io.PrintStream;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import opt.OptimizationAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.DataSetDescription;
import shared.Instance;
import shared.SumOfSquaresError;
import func.nn.NeuralNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;

public class Experiment {
	
	private final PrintStream outputStream;
	private final BackPropagationNetworkFactory networkFactory;
	private final int numHiddenNodes;
	private final int numIterations;
	private final DataSet trainingData;
	private final DataSet testingData;
	private final int numRestarts;

	public Experiment(PrintStream outputStream, BackPropagationNetworkFactory networkFactory,
			int numHiddenNodes, int numIterations, int numRestarts, DataSet trainingData, DataSet testingData) {
		this.outputStream = outputStream;
		this.networkFactory = networkFactory;
		this.numHiddenNodes = numHiddenNodes;
		this.numIterations = numIterations;
		this.numRestarts = numRestarts;
		this.trainingData = trainingData;
		this.testingData = testingData;
	}
	
	public void runWithRestarts(AlgorithmFactory<OptimizationAlgorithm> algorithmFactory) {
		long start = System.currentTimeMillis();
		NetworkPerformance best = IntStream.range(0, numRestarts).parallel()
				.mapToObj(i -> runOnceAndPrint(algorithmFactory))
				.max(NetworkPerformance::compareByTestingPerf).get();
		long end = System.currentTimeMillis();
		
		System.out.println("Best Training: " + best.numTrainingCorrect);
		System.out.println("Best Testing: " + best.numTestingCorrect);
		
		long time = (end - start) / 1000;
		outputStream.println(time);
		System.out.println("Time: " + time + " seconds");
	}
	
	private NetworkPerformance runOnceAndPrint(AlgorithmFactory<OptimizationAlgorithm> algorithmFactory) {
		NetworkPerformance perf = runOnce(algorithmFactory, this::optimumInterpreter);
		printResult(perf.numTrainingCorrect, perf.numTestingCorrect);
		return perf;
	}
	
	public void runGenetic(AlgorithmFactory<StandardGeneticAlgorithmImproved> algorithmFactory) {
		long start = System.currentTimeMillis();
		runOnce(algorithmFactory, this::geneticInterpreter);
		long end = System.currentTimeMillis();
		long time = (end - start) / 1000;
		System.out.println("Time: " + time + " seconds");
		outputStream.println(time);
	}

	private <A extends OptimizationAlgorithm> NetworkPerformance runOnce(AlgorithmFactory<A> algorithmFactory,
			BiFunction<NeuralNetwork, A, NetworkPerformance> resultsInterpreter) {
		NeuralNetwork network = createNetwork();
		NeuralNetworkOptimizationProblem problem = new NeuralNetworkOptimizationProblem(trainingData, network, new SumOfSquaresError());
		A algorithm = algorithmFactory.apply(problem);
		
		runAlgorithm(algorithm);
		return resultsInterpreter.apply(network, algorithm);
	}
	
	private NetworkPerformance optimumInterpreter(NeuralNetwork network, OptimizationAlgorithm algorithm) {
		return evaluatePerformance(network, algorithm.getOptimal());
	}
	
	private NetworkPerformance geneticInterpreter(NeuralNetwork network, StandardGeneticAlgorithmImproved algorithm) {
		Instance[] instances = algorithm.getPopulation();
		NetworkPerformance best = evaluatePerformance(network, instances[0]);
		for (int i = 1; i < instances.length; i++) {
			NetworkPerformance perf = evaluatePerformance(network, instances[i]);
			if (NetworkPerformance.compareByTestingPerf(perf, best) > 0) {
				best = perf;
			}
		}
		
		return best;
	}
	
	private NetworkPerformance evaluatePerformance(NeuralNetwork network, Instance instance) {
		network.setWeights(instance.getData());
		int numTrainingCorrect = countNumCorrect(network, trainingData);
		int numTestingCorrect = countNumCorrect(network, testingData);
		
		printResult(numTrainingCorrect, numTestingCorrect);
		
		return new NetworkPerformance(numTrainingCorrect, numTestingCorrect);	
	}
	
	private synchronized void printResultToFile(int numTrainingCorrect, int numTestingCorrect) {
		outputStream.println(numTrainingCorrect + "," + numTestingCorrect);
	}
	
	private void printResult(int numTrainingCorrect, int numTestingCorrect) {
		printResultToFile(numTrainingCorrect, numTestingCorrect);

		System.out.println("Training Correct: " + numTrainingCorrect);
		System.out.println("Testing Correct: " + numTestingCorrect);
		System.out.println();
	}
	
	public static int countNumCorrect(NeuralNetwork network, DataSet data) {
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
