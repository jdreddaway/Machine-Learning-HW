package experiment;

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
}
