package experiments;
import java.util.function.Function;

import shared.DataSet;
import shared.Instance;


public class Evaluator {

	public boolean[] evaluate(DataSet data, Function<Instance, Instance> valueFn) {
		boolean[] results = new boolean[data.size()];

		for (int i = 0; i < data.size(); i++) {
			Instance instance = data.get(i);
			boolean correctCategory = instance.getLabel().getBoolean();
			boolean valuedCategory = valueFn.apply(instance).getBoolean();
			
			results[i] = correctCategory == valuedCategory;
		}
		
		return results;
	}
}
