package experiment;

import java.util.function.Function;

import rl.MarkovDecisionProcess;
import rl.MazeMarkovDecisionProcess;
import rl.Policy;
import rl.PolicyLearner;
import shared.Trainer;

public class PolicyCalculator {
	
	private final Function<Double, ? extends MazeMarkovDecisionProcess> mazeFactory;
	private final Function<MarkovDecisionProcess, ? extends PolicyLearner> learnerFactory;
	private final Function<PolicyLearner, ? extends Trainer> trainerFactory;

	public PolicyCalculator(Function<Double, ? extends MazeMarkovDecisionProcess> mazeFactory,
			Function<MarkovDecisionProcess, ? extends PolicyLearner> learnerFactory,
			Function<PolicyLearner, ? extends Trainer> trainerFactory) {
		this.mazeFactory = mazeFactory;
		this.learnerFactory = learnerFactory;
		this.trainerFactory = trainerFactory;
	}

	public Policy calcPolicy(double reward) {
		MazeMarkovDecisionProcess mdp = mazeFactory.apply(reward);
		PolicyLearner iter = learnerFactory.apply(mdp);
		Trainer trainer = trainerFactory.apply(iter);
		trainer.train();
		Policy policy = iter.getPolicy();
		
		/*
		MazeMarkovDecisionProcessVisualization visualizer = new MazeMarkovDecisionProcessVisualization(mdp);
		System.out.println(visualizer.toString(policy));
		*/

		return policy;
	}
}
