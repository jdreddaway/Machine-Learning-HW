package nnet_experiments;

import java.util.function.Function;

import opt.OptimizationAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;

public interface AlgorithmFactory<A extends OptimizationAlgorithm> extends Function<NeuralNetworkOptimizationProblem, A> {
}
