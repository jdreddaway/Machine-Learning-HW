package dimension_reduction;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import data.DataSetPrinter;
import shared.DataSet;
import shared.filt.RandomizedProjectionFilter;

public class RandomizedProjectionReducer extends RandomizedProjectionFilter implements DimensionReducer {

	private final int componentsOut;

	public RandomizedProjectionReducer(int componentsOut, int componentsIn) {
		super(componentsOut, componentsIn);
		this.componentsOut = componentsOut;
	}

	@Override
	public void printStatistics(String outputPath, DataSet reducedData) throws FileNotFoundException {
		try (PrintStream printer = new PrintStream(outputPath)) {
			printer.println(this.getClass().getSimpleName());
			printer.println(componentsOut);
			
			DataSet copiedData = reducedData.copy();
			this.reverse(copiedData);
			new DataSetPrinter().printData(printer, copiedData);
		}
	}

}
