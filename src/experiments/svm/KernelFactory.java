package experiments.svm;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.svm.Kernel;


public interface KernelFactory {

	public Kernel createKernel();
	
	public void serialize(PrintStream writer);
}
