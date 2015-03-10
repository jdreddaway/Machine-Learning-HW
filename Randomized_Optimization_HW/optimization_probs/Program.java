package optimization_probs;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.Stream;

import opt.HillClimbingProblem;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.MIMIC;
import shared.Instance;

public class Program {
	
	/*
	 * Randomized Hill Climbing
	 * 	ContinuousPeaks
	 * 	CountOnes
	 * 	FlipFlop
	 * 	FourPeaks
	 * 	Knapsack
	 * 	TravelingSalesman
	 * 
	 * Simulated Annealing
	 * 	ContinuousPeaks
	 * 	CountOnes
	 * 	FlipFlop
	 * 	FourPeaks
	 * 	Knapsack
	 * 	TravelingSalesman
	 * 
	 * Genetic
	 * 	ContinuousPeaks
	 * 	CountOnes
	 * 	FlipFlop
	 * 	FourPeaks
	 * 	Knapsack
	 * 	TravelingSalesman
	 * 
	 * MIMIC:
	 * 	ContinuousPeaks
	 * 	CountOnes
	 * 	FlipFlop
	 * 	Knapsack
	 * 	TravelingSalesman
	 * 
	 */
	
	public static void main(String[] args) {
		AllTypesOptimizationProblemSupplier[] problems = createProblems();
		Collection<AlgorithmRunnerFactory> algoRunners = createAlgoRunnerFactories();
		
		Stream.of(problems).parallel().forEach(probSupplier -> runAlgorithmsOnProblem(probSupplier, algoRunners));
	}
	
	private static void runAlgorithmsOnProblem(
			AllTypesOptimizationProblemSupplier problemSupplier, Collection<AlgorithmRunnerFactory> runnerFactories) {
		for (AlgorithmRunnerFactory each : runnerFactories) {
			AlgorithmRunner algoRunner = each.apply(problemSupplier);
			Stream<Instance> results = algoRunner.get();

			String outputFilepath = "output/" + problemSupplier.problemName + "_" + algoRunner.algoName + ".csv";
			try (PrintStream writer = new PrintStream(outputFilepath)) {
				results.forEach((instance) -> {
					double perf = problemSupplier.evalFn.value(instance);
					writer.println(perf);
				});
			} catch (FileNotFoundException e) {
				throw new RuntimeException("Could not open " + outputFilepath);
			}
			
			System.out.println("Finished " + outputFilepath);
		}
	}
	
	private static AllTypesOptimizationProblemSupplier[] createProblems() {
		return new AllTypesOptimizationProblemSupplier[] {
				AllTypesOptimizationProblemSupplier.createContinuousPeaksProblem(),
				AllTypesOptimizationProblemSupplier.createCountOnesProblem(),
				AllTypesOptimizationProblemSupplier.createFlipFlopProblem(),
				AllTypesOptimizationProblemSupplier.createKnapsackProblem(),
				//AllTypesOptimizationProblemSupplier.createTravelingSalesmanProblem() TODO uncomment
		};
	}
	
	private static Collection<AlgorithmRunnerFactory> createAlgoRunnerFactories() {
		Collection<AlgorithmRunnerFactory> algoRunnerFactories = new ArrayList<>();

		int numSamples = 1000;
		int numIterations = 3000;
		
		algoRunnerFactories.add(prob -> new WithRestartsRunner(
						"RandomizedHillClimbing",
						prob,
						p -> new RandomizedHillClimbing(p.hillClimbingProb.get()),
						numSamples,
						numIterations));

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
		algoRunnerFactories.add(prob -> new WithRestartsRunner(
						"SimulatedAnnealing",
						prob,
						p -> new SimulatedAnnealing(initialTemp, coolingRate, prob.hillClimbingProb.get()),
						numSamples,
						numIterations));
		
		double matePercentage = 0.55;
		double mutatePercentage = 0.15;
		int numToMate = (int) (matePercentage * numSamples);
		int numToMutate = (int) (mutatePercentage * numSamples);
		algoRunnerFactories.add(prob -> new IterationRunner(
				"Genetic",
				new StandardGeneticAlgorithm(numSamples, numToMate, numToMutate, prob.geneticProb.get()),
				numIterations));
		
		int numToKeep = 300;
		algoRunnerFactories.add(prob -> new IterationRunner(
				"MIMIC",
				new MIMIC(numSamples, numToKeep, prob.probabilisticProb.get()),
				numIterations));
		
		return algoRunnerFactories;
	}
}
