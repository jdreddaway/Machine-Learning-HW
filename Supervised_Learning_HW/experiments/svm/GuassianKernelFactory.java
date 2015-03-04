package experiments.svm;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.svm.Kernel;
import func.svm.RBFKernel;


public class GuassianKernelFactory implements KernelFactory {
	
	private double sigma;

	public GuassianKernelFactory(double sigma) {
		this.sigma = sigma;
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print("Guassian,");
		writer.print(sigma);
	}

	@Override
	public Kernel createKernel() {
		return new RBFKernel(sigma);
	}
}
