package nnet_experiments;

public class NetworkPerformance {

	public final int numTrainingCorrect;
	public final int numTestingCorrect;

	public NetworkPerformance(int numTrainingCorrect, int numTestingCorrect) {
		this.numTrainingCorrect = numTrainingCorrect;
		this.numTestingCorrect = numTestingCorrect;
		
	}
	
	public static int compareByTestingPerf(NetworkPerformance perf1, NetworkPerformance perf2) {
		return perf1.numTestingCorrect - perf2.numTestingCorrect;
	}
	
	public static int compareByTrainingPerf(NetworkPerformance perf1, NetworkPerformance perf2) {
		return perf1.numTrainingCorrect - perf2.numTrainingCorrect;
	}
}
