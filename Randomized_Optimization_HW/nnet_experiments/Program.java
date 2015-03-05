package nnet_experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.function.Function;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.StandardGeneticAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import data.UciDataReader;
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

		//expFactory.runExperiment("output/random_hillclimbing.out", prob -> new RandomizedHillClimbing(prob));

		double initialTemp = 1E11;
		double coolingExponent = .95;
		expFactory.runWithRestarts("output/simulated_annealing.out",
				prob -> new SimulatedAnnealing(initialTemp, coolingExponent, prob));

		double matePercentage = 0.55;
		double mutatePercentage = 0.15;
		int populationSize = numRestarts;
		int numToMate = (int) (matePercentage * populationSize);
		int numToMutate = (int) (mutatePercentage * populationSize);
		expFactory.runWithRestarts("output/genetic.out",
				prob -> new StandardGeneticAlgorithm(populationSize, numToMate, numToMutate, prob));
	}
}
