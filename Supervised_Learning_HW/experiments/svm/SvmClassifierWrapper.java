package experiments.svm;
import java.io.PrintStream;
import java.io.PrintWriter;

import data.filter.NormalizingFilter;
import dist.DiscreteDistribution;
import dist.Distribution;
import experiments.Classifier;
import experiments.Evaluator;
import func.SimpleSupportVectorMachineClassifier;
import func.svm.Kernel;
import func.svm.LinearKernel;
import func.svm.SequentialMinimalOptimization;
import func.svm.SupportVectorMachine;
import shared.DataSet;
import shared.DataSetDescription;
import shared.Instance;
import shared.filt.DiscreteToBinaryFilter;


public class SvmClassifierWrapper implements Classifier {
	
	private final Evaluator evaluator;
	private final KernelFactory kernelFactory;
	private final double c;

    private SupportVectorMachine svm;
	private NormalizingFilter filter;
	
	public SvmClassifierWrapper(Evaluator evaluator, double c, KernelFactory kernelFactory) {
		this.evaluator = evaluator;
		this.c = c;
		this.kernelFactory = kernelFactory;
	}

	@Override
	public void trainUsing(DataSet data) {
		estimate(data);
	}

	@Override
	public boolean[] evaluate(DataSet data) {
		return evaluator.evaluate(data, this::value);
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print(c);
		writer.print(',');
		kernelFactory.serialize(writer);
	}
    
    /**
     * Make a new svm classifier
     *
    public SimpleSupportVectorMachineClassifier() {
        this(1, new LinearKernel());
    }
    */

    /**
     * @param data Must be filtered first. Suggest using NormalizingFilter
     * @see func.FunctionApproximater#estimate(shared.DataSet)
     */
    public void estimate(DataSet data) {
    	/* This is what was originally here, but it creates a TON of features (i.e. attributes)
        DiscreteToBinaryFilter dtbf = new DiscreteToBinaryFilter();
        dtbf.filter(set);
        */
    	filter = new NormalizingFilter(new DataSetDescription(data));
    	filter.filter(data);

    	Kernel kernel = kernelFactory.createKernel();
        SequentialMinimalOptimization smo = new SequentialMinimalOptimization(data, kernel, c);
        smo.train();
        svm = smo.getSupportVectorMachine();
    }

    /**
     * @see func.FunctionApproximater#value(shared.Instance)
     */
    public Instance value(Instance instance) {
    	filter.filter(instance);
        return svm.value(instance);
    }
    
    /**
     * @see experiments.Classifier#classDistribution(shared.Instance)
     */
    public Distribution distributionFor(Instance data) {
        Instance v = value(data);
        double[] p = new double[2];
        p[v.getDiscrete()] = 1;
        return new DiscreteDistribution(p);
    }


	/*
	 * WHAT TO REPORT:
	 *  kernal, exponent or gamma, train %, test %, train time, test time
	 */
}
