package dimension_reduction;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import data.DataSetPrinter;
import shared.DataSet;
import shared.filt.PrincipalComponentAnalysis;
import util.linalg.Matrix;


public class PcaReducer extends PrincipalComponentAnalysis implements DimensionReducer {

	public PcaReducer(DataSet set) {
		super(set);
	}
	
	public PcaReducer(DataSet set, double percentVarianceToKeep) {
		super(set, percentVarianceToKeep);
	}

	@Override
	public void printStatistics(String outputPath, DataSet reducedData) throws FileNotFoundException {
		try (PrintStream printer = new PrintStream(outputPath)) {
			printer.println(this.getClass().getSimpleName());

			Matrix eigenvalues = this.getEigenValues();
			printer.print(eigenvalues.get(0, 0));
			for (int i = 1; i < eigenvalues.m(); i++) {
				printer.print(',');
				printer.print(eigenvalues.get(i, i));
			}
			printer.println();
			
			DataSet copiedData = reducedData.copy();
			this.reverse(copiedData);
			new DataSetPrinter().printData(printer, copiedData);
		}
	}

}
