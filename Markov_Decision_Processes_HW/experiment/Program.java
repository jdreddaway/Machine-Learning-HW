package experiment;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;
import java.util.function.Supplier;

import rl.MarkovDecisionProcess;
import rl.MazeMarkovDecisionProcessVisualization;
import rl.Policy;
import rl.PolicyIteration;
import rl.PolicyLearner;
import rl.ValueIteration;
import shared.ConvergenceTrainer;
import shared.Trainer;
import maze.NegativeMazeMDP;

public class Program {

	public static void main(String[] args) throws FileNotFoundException {
		File mazeFile = new File("mdp/cliffs.txt");
		double motionFailureProbability = .1;
		double trapCost = 10000000;
		double goalReward = 1;
		double gamma = 0.95;
		char[][] maze = NegativeMazeMDP.loadMaze(mazeFile);

		Function<Double, NegativeMazeMDP> mdpCreator =
				(reward) -> NegativeMazeMDP.createMaze(maze, motionFailureProbability, trapCost, reward);
		Function<MarkovDecisionProcess, PolicyLearner> learnerFactory = (mdp) -> new ValueIteration(gamma, mdp);
		Function<PolicyLearner, Trainer> trainerFactory = (learner) -> new ConvergenceTrainer(learner);
		PolicyCalculator policyCalculator = new PolicyCalculator(mdpCreator, learnerFactory, trainerFactory);
		//PolicyLearner iter = new PolicyIteration(gamma, mdp);
		
		NegativeMazeMDP sampleMdp = mdpCreator.apply(goalReward);
		Policy policy = policyCalculator.calcPolicy(goalReward);
		MazeMarkovDecisionProcessVisualization visualizer = new MazeMarkovDecisionProcessVisualization(sampleMdp);
		System.out.println(visualizer.toString(policy));
	}
	
	/**
	 * 
	 * @param mazeFactory (reward) -> maze
	 * @param learnerFactory
	 * @param trainerFactory 
	 * @return
	 */
	private static List<PolicyRange> findPoliciesWithDifferentRewards(
			PolicyCalculator policyFinder, Distinguisher<Policy> policyDistinguisher) {
		List<PolicyRange> policies = new ArrayList<>();
		final double initialIncrease = 16;
		
		double absoluteMaxReward = Double.MAX_VALUE;
		Policy absoluteUpperPolicy = policyFinder.calcPolicy(absoluteMaxReward);

		double lowerReward = 1;
		Policy lowerPolicy = policyFinder.calcPolicy(lowerReward);
		PolicyRange lowerRange = new PolicyRange(lowerReward, lowerReward, lowerPolicy);

		double upperMaxReward = 0;

		while (!policyDistinguisher.areEqual(lowerRange.policy, absoluteUpperPolicy)) {
			double increase = initialIncrease;
			upperMaxReward = lowerRange.maxReward + increase;

			Policy upperPolicy = policyFinder.calcPolicy(upperMaxReward);

			while (policyDistinguisher.areEqual(lowerRange.policy, upperPolicy)) {
				lowerRange.maxReward = upperMaxReward;
				upperMaxReward = upperMaxReward + 2 * increase;

				upperPolicy = policyFinder.calcPolicy(upperMaxReward);
			}
			
			PolicyRange upperRange = new PolicyRange(upperMaxReward, upperMaxReward, upperPolicy);
			List<PolicyRange> newPolicyRanges = findPolicyRanges(
					policyFinder, policyDistinguisher, lowerRange, upperRange);
			policies.addAll(newPolicyRanges);
			
			lowerRange = upperRange;
		}
		
		lowerRange.maxReward = absoluteMaxReward;
		policies.add(lowerRange);
		
		return policies;
	}
	
	/**
	 * 
	 * @param mazeFactory
	 * @param learnerFactory
	 * @param policyDistinguisher
	 * @param trainerFactory
	 * @param lowerPolicy
	 * @param upperPolicy Fields modified as a side-effect
	 * @return A list of policies between lowerPolicy (inclusive) and upperPolicy (exclusive)
	 */
	private static List<PolicyRange> findPolicyRanges(
			PolicyCalculator policyCalculator, Distinguisher<Policy> policyDistinguisher,
			PolicyRange lowerPolicy, PolicyRange upperPolicy) {
		List<PolicyRange> policies = new ArrayList<>();

		double middleReward = (upperPolicy.minReward + lowerPolicy.maxReward) / 2;
		while (upperPolicy.minReward - lowerPolicy.maxReward > 1) {
			Policy middle = policyCalculator.calcPolicy(middleReward);

			if (policyDistinguisher.areEqual(middle, lowerPolicy.policy)) {
				lowerPolicy.maxReward = middleReward;
				middleReward = (upperPolicy.minReward + middleReward) / 2;
			} else if (policyDistinguisher.areEqual(middle, upperPolicy.policy)) {
				upperPolicy.minReward = middleReward;
				middleReward = (middleReward + lowerPolicy.maxReward) / 2;
			} else {
				PolicyRange middleRange = new PolicyRange(middleReward, middleReward, middle);
				List<PolicyRange> bottomPolicies = findPolicyRanges(policyCalculator,
						policyDistinguisher, lowerPolicy, middleRange);

				policies.addAll(bottomPolicies);
				lowerPolicy = middleRange;

				middleReward = (lowerPolicy.maxReward + upperPolicy.minReward) / 2;
			}
		}
		
		policies.add(lowerPolicy);
		System.out.println("Found new policy shift after " + lowerPolicy.maxReward);

		return policies;
	}
	
	private static boolean policiesAreEqual(Policy p1, Policy p2) {
		int[] p1Actions = p1.getActions();
		int[] p2Actions = p2.getActions();
		return Arrays.equals(p1Actions, p2Actions);
	}
}
