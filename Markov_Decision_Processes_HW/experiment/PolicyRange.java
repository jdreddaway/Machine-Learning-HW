package experiment;

import rl.MazeMarkovDecisionProcessVisualization;
import rl.Policy;

public class PolicyRange {

	public double minReward;
	public double maxReward;
	public Policy policy;
	
	public PolicyRange(double minReward, double maxReward, Policy policy) {
		this.minReward = minReward;
		this.maxReward = maxReward;
		this.policy = policy;
	}
	
	public String toString(MazeMarkovDecisionProcessVisualization visualizer) {
		String ret = minReward + " - " + maxReward + "\n";
		return ret + visualizer.toString(policy);
	}
}
