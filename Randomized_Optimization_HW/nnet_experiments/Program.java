package nnet_experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import opt.EvaluationFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkEvaluationFunction;
import opt.example.NeuralNetworkWeightDistribution;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import shared.DataSet;
import shared.DataSetDescription;
import shared.ErrorMeasure;
import shared.SumOfSquaresError;
import data.UciDataReader;
import dist.DiscreteDependencyTree;
import dist.Distribution;
import func.nn.NeuralNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;

public class Program {
	
	public static void main(String[] args) throws FileNotFoundException {
		int numHiddenNodes = 15;
		int numIterations = 3000;
		int numRestarts = 1000;

		String trainingPath = "higgs/training1000.csv";
		String testingPath = "higgs/testing1000.csv";
		DataSet trainingData = new UciDataReader(new File(trainingPath)).read();
		DataSet testingData = new UciDataReader(new File(testingPath)).read();

		ExperimentRunner expFactory = new ExperimentRunner(numHiddenNodes, numIterations, numRestarts, trainingData, testingData);

		//expFactory.runWithRestarts("output/random_hillclimbing.out", prob -> new RandomizedHillClimbing(prob));

		double initialTemp = 1E11;

		/* How to calculate good cooling rate:
		 * 
		 * Final % chance of acceptance = e^(-worse / (initialTemp * coolingRate^numIterations))
		 * worse is how much worse the neighbor is
		 * 
		 * for 0.01% = e^(-1 / (1E11 * coolingRate^3000))
		 * coolingRate = ~0.99
		 */
		double coolingRate = .99;

		/*
		expFactory.runWithRestarts("output/simulated_annealing.out",
				prob -> new SimulatedAnnealing(initialTemp, coolingRate, prob));

		double matePercentage = 0.55;
		double mutatePercentage = 0.15;
		int populationSize = numRestarts;
		int numToMate = (int) (matePercentage * populationSize);
		int numToMutate = (int) (mutatePercentage * populationSize);
		expFactory.runOnce("output/genetic.out",
				prob -> new StandardGeneticAlgorithm(populationSize, numToMate, numToMutate, prob));
		*/
		
		//runMimicExperiments(numHiddenNodes, numIterations, trainingData, testingData);
	}

	private static void runMimicExperiments(int numHiddenNodes,
			int numIterations, DataSet trainingData, DataSet testingData)
			throws FileNotFoundException {
		BackPropagationNetworkFactory networkFactory = new BackPropagationNetworkFactory();
		ErrorMeasure errorMeasure = new SumOfSquaresError();
		int numAttributes = new DataSetDescription(trainingData).getAttributeCount();
		int[] layerNodeCounts = { numAttributes, numHiddenNodes, 1 };
		NeuralNetwork network = networkFactory.createClassificationNetwork(layerNodeCounts);
		EvaluationFunction evalFn = new NeuralNetworkEvaluationFunction(
				network, trainingData, errorMeasure);
		Distribution initialDistrib = new NeuralNetworkWeightDistribution(network.getLinks().size());
		ProbabilisticOptimizationProblem probabilisticProblem = new GenericProbabilisticOptimizationProblem(
				evalFn, initialDistrib, initialDistrib);

		MIMIC mimicAlgo = new MIMIC(1000, 300, probabilisticProblem);
		runMimic(mimicAlgo, numIterations, "output/mimic300.out", network, trainingData, testingData);

		mimicAlgo = new MIMIC(1000, 500, probabilisticProblem);
		runMimic(mimicAlgo, numIterations, "output/mimic500.out", network, trainingData, testingData);

		mimicAlgo = new MIMIC(1000, 700, probabilisticProblem);
		runMimic(mimicAlgo, numIterations, "output/mimic700.out", network, trainingData, testingData);
	}
	
	private static void runMimic(MIMIC algo, int numIterations, String outputFilepath,
			NeuralNetwork network, DataSet trainingData, DataSet testData) throws FileNotFoundException {
		for (int i = 0; i < numIterations; i++) {
			algo.train();
		}
		
		network.setWeights(algo.getOptimal().getData());
		int numTrainingCorrect = Experiment.countNumCorrect(network, trainingData);
		int numTestCorrect = Experiment.countNumCorrect(network, testData);
		System.out.println(numTrainingCorrect + "," + numTestCorrect);

		try (PrintStream printer = new PrintStream(outputFilepath)) {
			printer.println(numTrainingCorrect + "," + numTestCorrect);
		}
	}
}
