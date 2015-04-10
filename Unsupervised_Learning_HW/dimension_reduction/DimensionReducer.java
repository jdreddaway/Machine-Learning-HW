package dimension_reduction;
import java.io.FileNotFoundException;

import shared.DataSet;
import shared.filt.ReversibleFilter;

public interface DimensionReducer extends ReversibleFilter {

	public void printStatistics(String outputPath, DataSet reducedData) throws FileNotFoundException;
}
