package optimization_probs;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Supplier;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.SwapNeighbor;
import opt.example.ContinuousPeaksEvaluationFunction;
import opt.example.CountOnesEvaluationFunction;
import opt.example.FourPeaksEvaluationFunction;
import opt.example.KnapsackEvaluationFunction;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.ProbabilisticOptimizationProblem;

public class AllTypesOptimizationProblemSupplier {
	public final Supplier<HillClimbingProblem> hillClimbingProb;
	public final Supplier<GeneticAlgorithmProblem> geneticProb;
	public final Supplier<ProbabilisticOptimizationProblem> probabilisticProb;
	public final String problemName;
	public final EvaluationFunction evalFn;
	
	public AllTypesOptimizationProblemSupplier(
			String problemName, Supplier<HillClimbingProblem> hcProb, Supplier<GeneticAlgorithmProblem> geneticProb,
			Supplier<ProbabilisticOptimizationProblem> probabilisticProb, EvaluationFunction evalFn) {
		this.problemName = problemName;
		hillClimbingProb = hcProb;
		this.geneticProb = geneticProb;
		this.probabilisticProb = probabilisticProb;
		this.evalFn = evalFn;
	}
	
	public static AllTypesOptimizationProblemSupplier createContinuousPeaksProblem() {
	    final int N = 60;
	    final int T = N / 10;

		int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        Supplier<HillClimbingProblem> hcp = () -> new GenericHillClimbingProblem(ef, odd, nf);
        Supplier<GeneticAlgorithmProblem> gap = () -> new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        Supplier<ProbabilisticOptimizationProblem> pop = () -> new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        return new AllTypesOptimizationProblemSupplier("ContinuousPeaks", hcp, gap, pop, ef);
	}
	
	public static AllTypesOptimizationProblemSupplier createCountOnesProblem() {
		final int N = 80;
		int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        Supplier<HillClimbingProblem> hcp = () -> new GenericHillClimbingProblem(ef, odd, nf);
        Supplier<GeneticAlgorithmProblem> gap = () -> new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        Supplier<ProbabilisticOptimizationProblem> pop = () -> new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        return new AllTypesOptimizationProblemSupplier("CountOnes", hcp, gap, pop, ef);
	}
	
	public static AllTypesOptimizationProblemSupplier createFlipFlopProblem() {
		final int N = 80;
	    final int T = N/10;

		int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        Supplier<HillClimbingProblem> hcp = () -> new GenericHillClimbingProblem(ef, odd, nf);
        Supplier<GeneticAlgorithmProblem> gap = () -> new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        Supplier<ProbabilisticOptimizationProblem> pop = () -> new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        return new AllTypesOptimizationProblemSupplier("FlipFlop", hcp, gap, pop, ef);
	}
	
	public static AllTypesOptimizationProblemSupplier createKnapsackProblem() {
		final Random random = new Random();
	    final int NUM_ITEMS = 40;
	    final int COPIES_EACH = 4;
	    final double MAX_WEIGHT = 50;
	    final double MAX_VOLUME = 50;
	    final double KNAPSACK_VOLUME = 
	         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;

		int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
         int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        Supplier<HillClimbingProblem> hcp = () -> new GenericHillClimbingProblem(ef, odd, nf);
        Supplier<GeneticAlgorithmProblem> gap = () -> new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        Supplier<ProbabilisticOptimizationProblem> pop = () -> new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        return new AllTypesOptimizationProblemSupplier("Knapsack", hcp, gap, pop, ef);
	}
	
	public static AllTypesOptimizationProblemSupplier createTravelingSalesmanProblem() {
		final int N = 50;

		Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding

        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);

        Supplier<HillClimbingProblem> hcp = () -> new GenericHillClimbingProblem(ef, odd, nf);
        Supplier<GeneticAlgorithmProblem> gap = () -> new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        // for mimic we use a sort encoding
        TravelingSalesmanEvaluationFunction efs = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        Distribution oddProb = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        Supplier<ProbabilisticOptimizationProblem> pop = () -> new GenericProbabilisticOptimizationProblem(efs, oddProb, df);
        
        //TODO is it a problem that MIMIC uses a different evaluation function?
        return new AllTypesOptimizationProblemSupplier("TravelingSalesman", hcp, gap, pop, ef);
	}
}
