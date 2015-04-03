package optimization_probs;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.StandardGeneticAlgorithm;
import optimization_probs.runners.AlgorithmRunner;
import optimization_probs.runners.AlgorithmRunnerFactory;
import optimization_probs.runners.IterationRunner;
import optimization_probs.runners.RunResult;
import optimization_probs.runners.WithParallelRestartsRunner;

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
	 * 	FourPeaks
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
			RunResult result = algoRunner.get();

			String outputFilepath = "output/" + problemSupplier.problemName + "_" + algoRunner.algoName + ".csv";
			String errorFilepath = "output/" + problemSupplier.problemName + "_" + algoRunner.algoName + "_error.csv";
			try (
					PrintStream writer = new PrintStream(outputFilepath);
					PrintStream errorWriter = new PrintStream(errorFilepath)
			) {
				DoubleStream perfs = result.evaluateInstances(problemSupplier.evalFn);
				perfs.sequential().forEach(perf -> writer.println(perf));
				
				result.printErrors(errorWriter);
			} catch (FileNotFoundException e) {
				throw new RuntimeException("Could not open " + outputFilepath);
			}
			
			System.out.println("Finished " + outputFilepath);
		}
	}
	
	private static AllTypesOptimizationProblemSupplier[] createProblems() {
		return new AllTypesOptimizationProblemSupplier[] {
				//AllTypesOptimizationProblemSupplier.createContinuousPeaksProblem(), 
				//AllTypesOptimizationProblemSupplier.createCountOnesProblem(), 
				//AllTypesOptimizationProblemSupplier.createFlipFlopProblem(), 
				AllTypesOptimizationProblemSupplier.createKnapsackProblem(),
				//AllTypesOptimizationProblemSupplier.createTravelingSalesmanProblem(),
				//AllTypesOptimizationProblemSupplier.createFourPeaksProblem() 
		};
	}
	
	private static Collection<AlgorithmRunnerFactory> createAlgoRunnerFactories() {
		Collection<AlgorithmRunnerFactory> algoRunnerFactories = new ArrayList<>();

		int numSamples = 1000;
		
		// Using just iterations
		int numIterations = 3000;
		Function<OptimizationAlgorithm, AlgorithmRunner> restartsIterationRunnerSupplier = 
				(algo) -> new IterationRunner("iter_" + algo.getClass().getSimpleName(), algo, numIterations);
		Function<OptimizationAlgorithm, AlgorithmRunner> noRestartsIterationRunnerSupplier = restartsIterationRunnerSupplier;

		// Using time to determine when to stop
		/*
		int totalTimeToRun = 60000;
		Function<OptimizationAlgorithm, AlgorithmRunner> restartsIterationRunnerSupplier =
				(algo) -> new TimedRunner(algo, totalTimeToRun / numSamples);
		Function<OptimizationAlgorithm, AlgorithmRunner> noRestartsIterationRunnerSupplier =
				(algo) -> new TimedRunner(algo, totalTimeToRun);
		*/
		
		algoRunnerFactories.add(prob -> new WithParallelRestartsRunner(
						"RandomizedHillClimbing",
						prob,
						p -> new RandomizedHillClimbing(p.hillClimbingProb.get()),
						numSamples,
						restartsIterationRunnerSupplier));

		double initialTemp = 1E11;

		/* How to calculate good cooling rate:
		 * 
		 * Final % chance of acceptance = e^(-worse / (initialTemp * coolingRate^numIterations))
		 * worse is how much worse the neighbor is
		 * 
		 * for 0.01% = e^(-1 / (1E11 * coolingRate^3000))
		 * coolingRate = ~0.99
		 */
		double coolingRate = .98;
		algoRunnerFactories.add(prob -> new WithParallelRestartsRunner(
						"SimulatedAnnealing",
						prob,
						p -> new SimulatedAnnealing(initialTemp, coolingRate, prob.hillClimbingProb.get()),
						numSamples,
						restartsIterationRunnerSupplier));
		

		double matePercentage = 0.55;
		double mutatePercentage = 0.15;
		int numToMate = (int) (matePercentage * numSamples);
		int numToMutate = (int) (mutatePercentage * numSamples);
		algoRunnerFactories.add(prob -> noRestartsIterationRunnerSupplier.apply(
					new StandardGeneticAlgorithm(numSamples, numToMate, numToMutate, prob.geneticProb.get())));
		
		int numToKeep = 300;
		algoRunnerFactories.add(prob -> noRestartsIterationRunnerSupplier.apply(
				new MimicImproved(numSamples, numToKeep, prob.probabilisticProb.get())));
		
		return algoRunnerFactories;
	}
}
