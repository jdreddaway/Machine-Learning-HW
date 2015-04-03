import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import shared.DataSet;
import data.UciDataReader;
import func.EMClusterer;
import func.FunctionApproximater;
import func.KMeansClusterer;


public class Program {

	public static void main(String[] args) throws IOException {
		UciDataReader dataReader = new UciDataReader(new File("higgs/training1000.csv"));
		DataSet data = dataReader.read();
		runClustering(k -> new KMeansClusterer(), data, "output/higgsk.csv");
		runClustering(k -> new EMClusterer(k, 1E-6, 1000), data, "output/higgsem.csv");
	}
	
	private static void runClustering(
			Function<Integer, FunctionApproximater> clustererCreator, DataSet data, String outputPath) throws IOException {
		List<ClusterResult> clusterResults = new ArrayList<>();
		for (int k = 2; k <= 10; k++) {
			FunctionApproximater clusterer = clustererCreator.apply(k);
			clusterer.estimate(data);
			clusterResults.add(ClusterResult.create(k, clusterer, data));
		}

		try (
			FileOutputStream outputStream = new FileOutputStream(outputPath, false);
			PrintStream printer = new PrintStream(outputStream);
		) {
			for (ClusterResult result : clusterResults) {
				result.print(printer);
				printer.println();
			}
		}
	}
	
	private void runDimensionReduction(DataSet data, String outputPath) {
		//TODO finish after done with clustering algos
	}
}
