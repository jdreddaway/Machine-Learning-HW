package experiments.svm;
import java.io.PrintStream;
import java.io.PrintWriter;

import func.svm.Kernel;
import func.svm.PolynomialKernel;


public class PolynomialKernelFactory implements KernelFactory {

	private double dotProductWeight;
	private double constant;
	private int exponent;

	/**
	 * 
	 * @param dotProductWeight suggest (0, 2]; 1 is normal
	 * @param constant suggest [0, 2]; {0, 1} are normal
	 * @param exponent suggest {1, 2, 3}
	 */
	public PolynomialKernelFactory(double dotProductWeight, double constant, int exponent) {
		this.dotProductWeight = dotProductWeight;
		this.constant = constant;
		this.exponent = exponent;
	}

	@Override
	public void serialize(PrintStream writer) {
		writer.print("Polynomial,");
		writer.print(dotProductWeight);
		writer.print(',');
		writer.print(constant);
		writer.print(',');
		writer.print(exponent);
	}

	@Override
	public Kernel createKernel() {
		return new PolynomialKernel(dotProductWeight, constant, exponent);
	}

}
