import java.util.List;

import shared.DataSet;


public interface Clusterer<ResultType extends ClusterResult> {

	public List<ResultType> createClusters(int minK, int maxK, DataSet dataset);
}
