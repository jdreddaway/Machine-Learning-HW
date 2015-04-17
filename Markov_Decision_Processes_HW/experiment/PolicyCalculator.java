package experiment;

import java.util.function.Function;

import maze.NegativeMazeMDP;
import rl.MarkovDecisionProcess;
import rl.Policy;
import rl.PolicyLearner;
import shared.Trainer;

public class PolicyCalculator {
	
	private final Function<Double, ? extends MarkovDecisionProcess> mazeFactory;
	private final Function<MarkovDecisionProcess, ? extends PolicyLearner> learnerFactory;
	private final Function<PolicyLearner, ? extends Trainer> trainerFactory;

	public PolicyCalculator(Function<Double, ? extends MarkovDecisionProcess> mazeFactory,
			Function<MarkovDecisionProcess, ? extends PolicyLearner> learnerFactory,
			Function<PolicyLearner, ? extends Trainer> trainerFactory) {
		this.mazeFactory = mazeFactory;
		this.learnerFactory = learnerFactory;
		this.trainerFactory = trainerFactory;
	}

	public Policy calcPolicy(double reward) {
		MarkovDecisionProcess mdp = mazeFactory.apply(reward);
		PolicyLearner iter = learnerFactory.apply(mdp);
		Trainer trainer = trainerFactory.apply(iter);
		trainer.train();
		Policy policy = iter.getPolicy();
		return policy;
	}
}
