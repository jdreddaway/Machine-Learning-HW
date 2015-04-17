package experiment;

import rl.MarkovDecisionProcess;
import rl.Policy;

public class OptimalPolicyDistinguisher implements Distinguisher<Policy> {
	
	private final MarkovDecisionProcess mdp;

	public OptimalPolicyDistinguisher(MarkovDecisionProcess mdp) {
		this.mdp = mdp;
	}

	@Override
	public boolean areEqual(Policy p1, Policy p2) {
		int currentState = mdp.sampleInitialState();
		while (!mdp.isTerminalState(currentState)) {
			int nextState1 = mdp.sampleState(currentState, p1.getAction(currentState));
			int nextState2 = mdp.sampleState(currentState, p2.getAction(currentState));
			
			if (nextState1 != nextState2) {
				return false;
			}
			
			currentState = nextState1;
		}

		return true;
	}

}
