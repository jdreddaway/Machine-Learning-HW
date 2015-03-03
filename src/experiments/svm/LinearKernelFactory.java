package experiments.svm;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.svm.Kernel;
import func.svm.LinearKernel;


public class LinearKernelFactory implements KernelFactory {

	@Override
	public void serialize(PrintStream writer) {
		writer.print("Linear");
	}

	@Override
	public Kernel createKernel() {
		return new LinearKernel();
	}

}
