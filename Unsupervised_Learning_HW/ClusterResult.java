import java.io.PrintStream;

import shared.DataSet;
import shared.Instance;
import func.FunctionApproximater;


public class ClusterResult {
	
	private int[] numInstancesPerCluster;

	public ClusterResult(int[] numInstancesPerCluster) {
		this.numInstancesPerCluster = numInstancesPerCluster;
	}

	public void print(PrintStream printer) {
		for (int value : numInstancesPerCluster) {
			printer.println(value);
		}
	}
	
	public static ClusterResult create(FunctionApproximater clusterer, DataSet data, int numClusters) {
		int[] numInstancesPerCluster = new int[numClusters];

		for (int i = 0; i < data.size(); i++) {
			Instance currentInstance = data.get(i);
			int cluster = clusterer.value(currentInstance).getDiscrete();

			numInstancesPerCluster[cluster]++;
		}
		
		return new ClusterResult(numInstancesPerCluster);
	}
	
	/*
	private double[][] calcDifferences(KMeansClusterResult other) {
		double[][] ret = new double[centers.length][other.centers.length];
		
		for (int m = 0; m < centers.length; m++) {
			for (int o = 0; o < other.centers.length; o++) {
				ret[m][o] = distMeasure.value(centers[m], other.centers[o]);
			}
		}
		
		return ret;
	}
	
	public void print(PrintStream printer) {
		printer.println(k);
		for (Instance each : centers) {
			printer.println(each);
		}
	}
	
	public void printDifferences(PrintStream printer, KMeansClusterResult other) {
		double[][] differences = calcDifferences(other);
		
		printer.println(k + " and " + other.k);
		
		for (int i = 0; i < differences.length; i++) {
			printer.print(differences[i][0]);
			
			for (int j = 1; j < differences[i].length; j++) {
				printer.print("," + differences[i][j]);
			}
			
			printer.println();
		}
	}
	 */
}
