package dimension_reduction;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import data.DataSetPrinter;
import shared.DataSet;
import shared.filt.IndependentComponentAnalysis;


public class IndependentComponentAnalysisReducer extends IndependentComponentAnalysis implements DimensionReducer {

	public IndependentComponentAnalysisReducer(DataSet dataSet) {
		super(dataSet);
	}

	@Override
	public void printStatistics(String outputPath, DataSet reducedData) throws FileNotFoundException {
		try (PrintStream printer = new PrintStream(outputPath)) {
			printer.println(this.getClass().getSimpleName());

			DataSet copiedData = reducedData.copy();
			this.reverse(copiedData);
			new DataSetPrinter().printData(printer, copiedData);
		}
	}

}
