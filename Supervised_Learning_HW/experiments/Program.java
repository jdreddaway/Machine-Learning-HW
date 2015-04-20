package experiments;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import data.extraction.DataExtracter;
import data.extraction.SimulationData;
import data.parsing.CategoryNotBinaryException;
import data.parsing.DataReorganizer;
import experiments.boosting.BoostingClassifierWrapper;
import experiments.boosting.DecisionStumpClassifierFactory;
import experiments.boosting.DecisionTreeFactory;
import experiments.boosting.FunctionApproximaterFactory;
import experiments.dtree.DecisionTreeClassifierWrapper;
import experiments.knn.KnnClassifierWrapper;
import experiments.nnet.BackPropagationTrainerFactory;
import experiments.nnet.BatchNetworkTrainerFactory;
import experiments.nnet.ConvergenceTrainerFactory;
import experiments.nnet.HalterTrainerFactory;
import experiments.nnet.NeuralNetworkClassifierWrapper;
import experiments.nnet.StochasticNetworkTrainerFactory;
import experiments.svm.GuassianKernelFactory;
import experiments.svm.KernelFactory;
import experiments.svm.LinearKernelFactory;
import experiments.svm.PolynomialKernelFactory;
import experiments.svm.SvmClassifierWrapper;
import shared.DistanceMeasure;
import shared.EuclideanDistance;
import func.dtree.GINISplitEvaluator;
import func.nn.activation.DifferentiableActivationFunction;
import func.nn.activation.LinearActivationFunction;
import func.nn.activation.LogisticSigmoid;
import func.nn.backprop.BasicUpdateRule;
import func.nn.backprop.StandardUpdateRule;
import func.nn.backprop.WeightUpdateRule;


public class Program {
	
	private static String[] trainingFiles = {
		"higgs/training250.csv",
		"higgs/training500.csv",
		"higgs/training750.csv",
		"higgs/training1000.csv",
	};
	
	private static String testFile = "higgs/testing1000.csv";
	private static int numFeatures = 28; // 109;

	public static void main(String[] args) throws IOException {
		//System.out.println(countNumDatapoints("adult/training.csv"));
		//extractData(1000, 1500, "resources/adult_processed.csv", 32561);
		//extractData(750, 1000, "resources/adult_processed.csv", 32561);
		//processAdultData();
		//runNeuralNetworkExperiment();
		//runDecisionTreeExperiment();
		//runBoostingExperiment();
		//runKnnExperiment();
		runSvmExperiment();
	}
	
	private static int countNumDatapoints(String filename) throws FileNotFoundException {
		int numLines = 0;
		try (Scanner scan = new Scanner(new File(filename))) {
			while (scan.hasNextLine()) {
				scan.nextLine();
				numLines++;
			}
		}
		
		return numLines;
	}
	
	private static void processAdultData() {
		DataReorganizer reorganizer = new DataReorganizer();
		try {
			reorganizer.reorganize("adult/adult.data", -1, null, "adult/training.csv");
		} catch (FileNotFoundException | CategoryNotBinaryException e) {
			throw new RuntimeException(e);
		}
	}

	private static void runKnnExperiment() throws FileNotFoundException {
		int[] kValues = { 3, 5, 9, 15, 25, 37 };
		boolean[] weightValues = {true, false};
		double[] ranges = { 0, 0.5, 1, 2, 4, 8, 16, 32, 64 };

		Collection<KnnClassifierWrapper> classifiers = createKnnClassifiers(kValues, weightValues, ranges);

		Experiment experiment = new Experiment(trainingFiles, testFile);
		experiment.runAll(classifiers, "knn_results2.csv");
	}

	private static Collection<KnnClassifierWrapper> createKnnClassifiers(
			int[] kValues, boolean[] weightValues, double[] ranges) {
		Permutator permutator = new Permutator(new int[] { kValues.length, weightValues.length, ranges.length });
		
		Evaluator evaluator = new Evaluator();
		DistanceMeasure distMeasure = new EuclideanDistance();
		Collection<KnnClassifierWrapper> classifiers = new ArrayList<>();
		while (permutator.hasNext()) {
			int[] indices = permutator.next();
			classifiers.add(new KnnClassifierWrapper(
					evaluator, kValues[indices[0]], weightValues[indices[1]], distMeasure, ranges[indices[2]]));
		}

		return classifiers;
	}

	private static void runSvmExperiment() throws FileNotFoundException {
		double[] cValues = { 0.125, 0.5, 2 };
		double[] sigmas = { 8, 2, 0.5, 0.125 };
		
		List<KernelFactory> kernels = createGuassianKernels(sigmas);
		
		double[] dotProductWeights = { 0.5, 1, 2 };
		double[] constants = { 0, 0.5, 1, 2 };
		int[] exponents = { 1, 2, 3};
		kernels.addAll(createPolynomialKernels(dotProductWeights, constants, exponents));
		
		kernels.add(new LinearKernelFactory());

		Collection<SvmClassifierWrapper> classifiers = createSvmClassifiers(cValues, kernels);
		
		Experiment experiment = new Experiment(trainingFiles, testFile);
		experiment.runAll(classifiers, "svm_results2.csv");
	}
	
	private static Collection<SvmClassifierWrapper> createSvmClassifiers(
			double[] cValues, List<KernelFactory> kernels) {
		Permutator permutater = new Permutator(new int[] { cValues.length, kernels.size() });
		Collection<SvmClassifierWrapper> classifiers = new ArrayList<>();
		Evaluator evaluater = new Evaluator();
		
		while (permutater.hasNext()) {
			int[] indices = permutater.next();
			classifiers.add(new SvmClassifierWrapper(evaluater, cValues[indices[0]], kernels.get(indices[1])));
		}
		
		return classifiers;
	}

	private static Collection<? extends KernelFactory> createPolynomialKernels(
			double[] dotProductWeights, double[] constants, int[] exponents) {
		Permutator permutater = new Permutator(new int[] {
				dotProductWeights.length, constants.length, exponents.length });

		Collection<KernelFactory> kernels = new ArrayList<>();
		while (permutater.hasNext()) {
			int[] indices = permutater.next();
			kernels.add(new PolynomialKernelFactory(
					dotProductWeights[indices[0]], constants[indices[1]], exponents[indices[2]]));
		}
		
		return kernels;
	}

	private static List<KernelFactory> createGuassianKernels(double[] sigmas) {
		List<KernelFactory> kernels = new ArrayList<>();
		
		for (double sigma : sigmas) {
			kernels.add(new GuassianKernelFactory(sigma));
		}

		return kernels;
	}

	private static void runBoostingExperiment() throws FileNotFoundException {
		int[] confidenceLevels = { 0, 1, 2, 3, 4 };
		//List<FunctionApproximaterFactory> functionApproximaterFactores = createDecisionTreeFactories(confidenceLevels);
		List<FunctionApproximaterFactory> functionApproximaterFactores = new ArrayList<>();
		functionApproximaterFactores.add(new DecisionStumpClassifierFactory());
		
		int[] sizes = { 10, 20, 30, 40 };
		Collection<BoostingClassifierWrapper> classifiers = createBoostingClassifiers(functionApproximaterFactores, sizes);
		
		Experiment experiment = new Experiment(trainingFiles, testFile);
		experiment.runAll(classifiers, "boosting_results1.csv");
	}
	
	private static Collection<BoostingClassifierWrapper> createBoostingClassifiers(
			List<FunctionApproximaterFactory> functionApproximaterFactories, int[] sizes) {
		Collection<BoostingClassifierWrapper> ret = new ArrayList<>();
		Permutator permutator = new Permutator(new int[] { functionApproximaterFactories.size(), sizes.length });
		Evaluator evaluator = new Evaluator();
		
		while (permutator.hasNext()) {
			int[] indices = permutator.next();
			ret.add(new BoostingClassifierWrapper(evaluator, sizes[indices[1]], functionApproximaterFactories.get(indices[0])));
		}
		
		return ret;
	}
	
	private static List<FunctionApproximaterFactory> createDecisionTreeFactories(int[] confidenceLevels) {
		List<FunctionApproximaterFactory> factories = new ArrayList<>();
		GINISplitEvaluator splitEvaluator = new GINISplitEvaluator();
		for (int confidence : confidenceLevels) {
			factories.add(new DecisionTreeFactory(splitEvaluator, confidence, true));
		}
		
		return factories;
	}

	private static void runNeuralNetworkExperiment() throws FileNotFoundException {
		DifferentiableActivationFunction[] activationFunctions = {
				new LinearActivationFunction(), new LogisticSigmoid()
		};
		WeightUpdateRule[] updateRules = {
				new BasicUpdateRule(), new StandardUpdateRule()
		};
		int[] numEpochs = { 500, 1000, 1500, 2000 };
		int[] numHiddenNodes = { 5, 15, 25 };
		double[] thresholds = { 1E-15, 1E-10, 1E-5 };
		
		Collection<NeuralNetworkClassifierWrapper> classifiers = createNeuralNetworkClassifiers(
				activationFunctions, updateRules, numEpochs, numHiddenNodes, thresholds);
		
		Experiment experiment = new Experiment(trainingFiles, testFile);
		experiment.runAll(classifiers, "nnet_results2.csv");
	}
	
	private static Collection<NeuralNetworkClassifierWrapper> createNeuralNetworkClassifiers(
			DifferentiableActivationFunction[] activationFunctions, WeightUpdateRule[] updateRules,
			int[] epochs, int[] numHiddenNodes, double[] thresholds) {
		Permutator permutator = new Permutator(new int[] {
				activationFunctions.length, updateRules.length, epochs.length, numHiddenNodes.length, thresholds.length
		});

		Evaluator evaluator = new Evaluator();

		Collection<NeuralNetworkClassifierWrapper> classifiers = new ArrayList<>();
		while (permutator.hasNext()) {
			int[] permutation = permutator.next();
			DifferentiableActivationFunction activationFunction = activationFunctions[permutation[0]];
			WeightUpdateRule updateRule = updateRules[permutation[1]];
			int numEpochs = epochs[permutation[2]];
			int numHidden = numHiddenNodes[permutation[3]];
			double threshold = thresholds[permutation[4]];
			
			BackPropagationTrainerFactory stochasticTrainer = new StochasticNetworkTrainerFactory(updateRule);
			BackPropagationTrainerFactory batchTrainer = new BatchNetworkTrainerFactory(updateRule);
			
			HalterTrainerFactory stochasticHalter = new ConvergenceTrainerFactory(
					stochasticTrainer, threshold, numEpochs);
			HalterTrainerFactory batchHalter = new ConvergenceTrainerFactory(
					batchTrainer, threshold, numEpochs);
			
			/*
			HalterTrainerFactory stochasticHalter = new FixedIterationTrainerFactory(numEpochs, stochasticTrainer);
			HalterTrainerFactory batchHalter = new FixedIterationTrainerFactory(numEpochs, batchTrainer);
			*/
			
			classifiers.add(new NeuralNetworkClassifierWrapper(
					evaluator, numHidden, numFeatures, activationFunction, stochasticHalter));
			classifiers.add(new NeuralNetworkClassifierWrapper(
					evaluator, numHidden, numFeatures, activationFunction, batchHalter));
		}
		
		return classifiers;
	}

	private static void runDecisionTreeExperiment() throws FileNotFoundException {
		Experiment experiment = new Experiment(trainingFiles, testFile);
		Evaluator evaluator = new Evaluator();

		try (PrintStream resultsWriter = new PrintStream("dtree_results2.csv")) {
			experiment.run(resultsWriter, new DecisionTreeClassifierWrapper(evaluator, true, true, 0));
			experiment.run(resultsWriter, new DecisionTreeClassifierWrapper(evaluator, true, true, 1));
			experiment.run(resultsWriter, new DecisionTreeClassifierWrapper(evaluator, true, true, 2));

			DateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");
			Date date = new Date();
			System.out.println(dateFormat.format(date));

			experiment.run(resultsWriter, new DecisionTreeClassifierWrapper(evaluator, true, true, 3));
			experiment.run(resultsWriter, new DecisionTreeClassifierWrapper(evaluator, true, true, 4));
			experiment.run(resultsWriter, new DecisionTreeClassifierWrapper(evaluator, true, false, 0));
		}
	}

	private static void extractData(int numTraining, int numTesting, String filename, int numDatapointsInFile) throws IOException {
		File datafile = new File(filename);
		DataExtracter extracter = new DataExtracter(new Random());
		try (
			InputStream is = new FileInputStream(datafile);
			InputStreamReader instrm = new InputStreamReader(is);
			BufferedReader reader = new BufferedReader(instrm);  
		) {
			SimulationData data = extracter.extractNumericData(reader, numDatapointsInFile, numTraining, numTesting);

			PrintWriter trainingWriter = new PrintWriter(new File("training" + numTraining + ".csv"));
			data.writeTrainingData(trainingWriter);
			trainingWriter.close();
			
			PrintWriter testingWriter = new PrintWriter(new File("testing" + numTesting + ".csv"));
			data.writeTestData(testingWriter);
			testingWriter.close();
		} catch (FileNotFoundException e) {
			System.out.println("Could not find data file.");
			e.printStackTrace();
		}
	}
}
