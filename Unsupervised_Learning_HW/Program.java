import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import shared.DataSet;
import shared.DataSetDescription;
import shared.Instance;
import shared.filt.IndependentComponentAnalysis;
import shared.filt.LinearDiscriminantAnalysis;
import shared.filt.PrincipalComponentAnalysis;
import shared.filt.RandomizedProjectionFilter;
import shared.filt.ReversibleFilter;
import util.linalg.DenseVector;
import util.linalg.Vector;
import data.DataSetPrinter;
import data.UciDataReader;
import dimension_reduction.DimensionReducer;
import dimension_reduction.IndependentComponentAnalysisReducer;
import dimension_reduction.LinearDiscriminantAnalysisReducer;
import dimension_reduction.PcaReducer;
import dimension_reduction.RandomizedProjectionReducer;
import dimension_reduction.InsignificantComponentAnalysisReducer;
import experiments.Evaluator;
import experiments.Results;
import experiments.nnet.BackPropagationTrainerFactory;
import experiments.nnet.BatchNetworkTrainerFactory;
import experiments.nnet.ConvergenceTrainerFactory;
import experiments.nnet.HalterTrainerFactory;
import experiments.nnet.NeuralNetworkClassifierWrapper;
import experiments.nnet.StochasticNetworkTrainerFactory;
import func.EMClusterer;
import func.FunctionApproximater;
import func.KMeansClusterer;import func.nn.activation.LinearActivationFunction;
import func.nn.backprop.BasicUpdateRule;
import func.nn.backprop.StandardUpdateRule;
import func.nn.backprop.WeightUpdateRule;



public class Program {

	public static void main(String[] args) throws IOException {
		//UciDataReader dataReader = new UciDataReader(new File("higgs/training1000.csv"));

		String dataName = "abalone";
		UciDataReader trainingReader = new UciDataReader(new File(dataName + "/initial.data"));
		DataSet trainingData = trainingReader.read();
		
		UciDataReader testingReader = new UciDataReader(new File(dataName + "/initial.test"));
		DataSet testingData = testingReader.read();

		String clusteringOutput = "output/" + dataName + "/clustered/";
		runClustering(k -> new KMeansClusterer(), trainingData, testingData, clusteringOutput, dataName);
		runClustering(k -> new EMClusterer(k, 1E-6, 1000), trainingData, testingData, clusteringOutput, dataName);

		String dimensionOutput = "output/" + dataName + "/";
//		runAllDimensionReduction(dataName, trainingData, testingData, dimensionOutput, 0.95);
//		
//		runClusteringOnFolder(k -> new KMeansClusterer(), dimensionOutput);
//		runClusteringOnFolder(k -> new EMClusterer(), dimensionOutput);
		
//		runNeuralNetOnFolder(dimensionOutput, "output/" + dataName + "/dimension_reduction.csv");
		runNeuralNetOnFolder(clusteringOutput, "output/" + dataName + "/clustered_nnet.csv");
		
//		try (PrintStream printer = new PrintStream("output/" + dataName + "/initial.csv")) {
//			SynchronizedPrinter syncPrinter = new SynchronizedPrinter(printer);
//			WeightUpdateRule updateRule = new BasicUpdateRule();
//			BackPropagationTrainerFactory stochasticTrainer = new StochasticNetworkTrainerFactory(updateRule);
//			HalterTrainerFactory stochasticHalter = new ConvergenceTrainerFactory(
//					stochasticTrainer, 1E-10, 1500);
//			NNetClassifierFactory classifierFactory = new NNetClassifierFactory(15, new LinearActivationFunction(), stochasticHalter);
//			File inputFile = new File(dataName + "/initial.data");
//			runNeuralNetwork(syncPrinter, classifierFactory, inputFile);
//		}
	}
	
	private static void runNeuralNetOnFolder(String folderPath, String outputFilepath) throws FileNotFoundException {
		File folder = new File(folderPath);
		
		WeightUpdateRule updateRule = new BasicUpdateRule();
		BackPropagationTrainerFactory stochasticTrainer = new StochasticNetworkTrainerFactory(updateRule);
		HalterTrainerFactory stochasticHalter = new ConvergenceTrainerFactory(
				stochasticTrainer, 1E-10, 1500);
		NNetClassifierFactory classifierFactory = new NNetClassifierFactory(15, new LinearActivationFunction(), stochasticHalter);

		Stream<File> dataFiles = Arrays.stream(folder.listFiles(Program::dataFileFilter)).parallel();
		try(PrintStream printer = new PrintStream(outputFilepath)) {
			SynchronizedPrinter syncPrinter = new SynchronizedPrinter(printer);
			dataFiles.forEach(input -> {
				runNeuralNetwork(syncPrinter, classifierFactory, input);
			});
		}
	}
	
	private static void runNeuralNetwork(SynchronizedPrinter printer, NNetClassifierFactory classifierFactory,
			File trainingFile) {
		UciDataReader trainingReader = new UciDataReader(trainingFile);
		DataSet trainingData;
		try {
			trainingData = trainingReader.read();
		} catch (FileNotFoundException e) {
			throw new RuntimeException("The file should not magically disappear", e);
		}
		
		String inputPath = trainingFile.getAbsolutePath();
		String testingFilepath = inputPath.substring(0, inputPath.lastIndexOf('.')) + ".test";

		UciDataReader testingReader = new UciDataReader(new File(testingFilepath));
		DataSet testingData;
		try {
			testingData = testingReader.read();
		} catch (FileNotFoundException fnfe) {
			throw new RuntimeException("Could not find the associated testing data.", fnfe);
		}
		
		int numFeatures = new DataSetDescription(trainingData).getAttributeCount();
		NeuralNetworkClassifierWrapper classifier = classifierFactory.create(numFeatures);
		
		classifier.trainUsing(trainingData);
		boolean[] trainingResults = classifier.evaluate(trainingData);
		boolean[] testResults = classifier.evaluate(testingData);
		Results results = new Results(0, 0, trainingResults, testResults);
		String dataName = trainingFile.getName();
		printer.println(dataName + "," + results.getTrainingAccuracy() + "," + results.getTestAccuracy());
	}
	
	/**
	 * 
	 * @param clustererSupplier
	 * @param folderPath The path to the input folder; must end in a '/'
	 * @throws IOException
	 */
	private static void runClusteringOnFolder(Function<Integer, FunctionApproximater> clustererSupplier, String folderPath) throws IOException {
		File folder = new File(folderPath);

		Stream<File> dataFiles = Arrays.stream(folder.listFiles(Program::dataFileFilter)).parallel();
		dataFiles.forEach(input -> {
			UciDataReader dataReader = new UciDataReader(input);
			DataSet trainingData;
			try {
				trainingData = dataReader.read();
			} catch (FileNotFoundException e) {
				throw new RuntimeException("The file should not magically disappear", e);
			}
			
			String inputPath = input.getAbsolutePath();
			String testingFilepath = inputPath.substring(0, inputPath.lastIndexOf('.')) + ".test";

			UciDataReader testingReader = new UciDataReader(new File(testingFilepath));
			DataSet testingData;
			try {
				testingData = testingReader.read();
			} catch (FileNotFoundException fnfe) {
				throw new RuntimeException("Could not find the associated testing data.", fnfe);
			}
			
			String fileNameWithoutExtension = input.getName().substring(0, input.getName().lastIndexOf('.'));
			String outputPath = folderPath + "clustered/";

			try {
				runClustering(clustererSupplier, trainingData, testingData, outputPath, fileNameWithoutExtension);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		});
	}
	
	private static boolean dataFileFilter(File file) {
		return file.isFile() && file.getName().endsWith(".data");
	}
	
	private static void runAllDimensionReduction(String dataName, DataSet trainingData, DataSet testingData,
			String outputFolder, double pcaPercentVarToKeep) throws FileNotFoundException {
		List<Supplier<DimensionReducer>> reducerSuppliers = new ArrayList<>();
		reducerSuppliers.add(() -> new IndependentComponentAnalysisReducer(trainingData));
		
		if (pcaPercentVarToKeep <= 1 && pcaPercentVarToKeep >= 0) {
			reducerSuppliers.add(() -> new PcaReducer(trainingData, pcaPercentVarToKeep));
			reducerSuppliers.add(() -> new InsignificantComponentAnalysisReducer(trainingData, pcaPercentVarToKeep));
		} else {
			reducerSuppliers.add(() -> new PcaReducer(trainingData));
			reducerSuppliers.add(() -> new InsignificantComponentAnalysisReducer(trainingData));
		}

		Stream<Supplier<DimensionReducer>> reducerSupplierStream = reducerSuppliers.parallelStream();
		reducerSupplierStream.forEach(reducerSupplier -> {
			DimensionReducer reducer = reducerSupplier.get();
			runCompleteDimensionReduction(trainingData, testingData, outputFolder, reducer);
		});
		runRandomizedProjection(dataName, trainingData, testingData);
	}

	private static void runCompleteDimensionReduction(DataSet trainingData, DataSet testingData, String outputFolder,
			DimensionReducer reducer) {
		String trainingOutputPath = outputFolder + reducer.getClass().getSimpleName() + ".data";
		String statisticsOutputPath = outputFolder + reducer.getClass().getSimpleName() + ".csv";
		try {
			DataSet reduced = runDimensionReduction(reducer, trainingData, trainingOutputPath);
			reducer.printStatistics(statisticsOutputPath, reduced);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		String testingOutputPath = outputFolder + reducer.getClass().getSimpleName() + ".test";
		try {
			runDimensionReduction(reducer, testingData, testingOutputPath);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	private static void runRandomizedProjection(String dataName, DataSet trainingData, DataSet testingData) throws FileNotFoundException {
		int numFeatures = new DataSetDescription(trainingData).getAttributeCount();
		
		for (int toKeep = 4; toKeep < numFeatures; toKeep++) {
			DimensionReducer reducer = new RandomizedProjectionReducer(toKeep, numFeatures);

			String trainingOutput = "output/" + dataName + "/randomized_projection_" + toKeep + ".data";
			String statisticsOutput = "output/" + dataName + "/randomized_projection_" + toKeep + ".csv";
			DataSet reduced = runDimensionReduction(reducer, trainingData, trainingOutput);
			reducer.printStatistics(statisticsOutput, reduced);

			String testingOutput = "output/" + dataName + "/randomized_projection_" + toKeep + ".test";
			runDimensionReduction(reducer, testingData, testingOutput);
		}
	}
	
	/**
	 * 
	 * @param clustererCreator
	 * @param data
	 * @param outputFolder The path of the output folder; must end in a '/'
	 * @param dataName The name of the data; used in the output filename
	 * @throws IOException
	 */
	private static void runClustering(
			Function<Integer, FunctionApproximater> clustererCreator, DataSet trainingData, DataSet testingData,
			String outputFolder, String dataName) throws IOException {
		List<ClusterResult> clusterResults = new ArrayList<>();
		FunctionApproximater clusterer = null;
		for (int k = 2; k <= 6; k++) {
			clusterer = clustererCreator.apply(k);
			clusterer.estimate(trainingData);
			
			String outputDataPath = outputFolder + dataName + "_" + clusterer.getClass().getSimpleName() + "_" + k + ".data";
			saveClusteredData(clusterer, k, trainingData, outputDataPath);

			String outputTestPath = outputFolder + dataName + "_" + clusterer.getClass().getSimpleName() + "_" + k + ".test";
			saveClusteredData(clusterer, k, testingData, outputTestPath);
			
			clusterResults.add(ClusterResult.create(clusterer, trainingData, k));
		}

		String analysisPath = outputFolder + dataName + "_" + clusterer.getClass().getSimpleName() + ".csv";
		printClusteringAnalysis(analysisPath, clusterResults);
	}
	
	private static void saveClusteredData(FunctionApproximater clusterer, int numClusters, DataSet data, String outputPath) throws FileNotFoundException {
		Instance[] newInstances = new Instance[data.size()];
		for (int i = 0; i < data.size(); i++) {
			Instance currentInstance = data.get(i);
			int cluster = clusterer.value(currentInstance).getDiscrete();

			newInstances[i] = (Instance) currentInstance.copy();
			appendClusterNumber(newInstances[i], cluster, numClusters);
		}
		
		DataSet newData = new DataSet(newInstances);
		try (PrintStream printer = new PrintStream(outputPath)) {
			new DataSetPrinter().printData(printer, newData);
		}
	}
	
	private static void appendClusterNumber(Instance instance, int clusterNumber, int numClusters) {
		int oldDataSize = instance.getData().size();
		Vector newData = new DenseVector(oldDataSize + numClusters);
		newData.set(0, instance.getData());

		double[] clusterData = new double[numClusters];
		clusterData[clusterNumber] = 1;
		Vector clusterVector = new DenseVector(clusterData);
		newData.set(oldDataSize, clusterVector);

		instance.setData(newData);
	}
	
	private static void printClusteringAnalysis(String outputPath, List<ClusterResult> results) throws FileNotFoundException, IOException {
		try (
			FileOutputStream outputStream = new FileOutputStream(outputPath, false);
			PrintStream printer = new PrintStream(outputStream);
		) {
			for (ClusterResult result : results) {
				result.print(printer);
				printer.println();
			}
		}
	}
	
	private static DataSet runDimensionReduction(DimensionReducer reducer, DataSet data, String outputPath) throws FileNotFoundException {
		DataSet dataCopy = data.copy();
		reducer.filter(dataCopy);
		
		try (PrintStream printer = new PrintStream(outputPath)) {
			new DataSetPrinter().printData(printer, dataCopy);
		}
		
		return dataCopy;
	}
}
