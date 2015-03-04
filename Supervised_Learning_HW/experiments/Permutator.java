package experiments;
import java.util.Arrays;


public class Permutator {

	private final int[] sizes;
	private int[] next;
	private boolean hasNext = true;

	/**
	 * 
	 * @param sizes all elements must be > 0
	 */
	public Permutator(int[] sizes) {
		this.sizes = sizes;
		next = new int[sizes.length];
	}
	
	public boolean hasNext() {
		return hasNext;
	}
	
	public int[] getNext() {
		int[] ret = Arrays.copyOf(next, next.length);
		
		boolean shouldUpdate = true;
		
		for (int i = 0; i < next.length && shouldUpdate; i++) {
			next[i]++;
			if (next[i] < sizes[i]) {
				shouldUpdate = false;
			} else {
				for (int j = 0; j <= i; j++) {
					next[j] = 0;
				}
			}
		}
		
		if (shouldUpdate) {
			hasNext = false;
		}
		
		return ret;
	}
	
	public int getCount() {
		int count = 1;
		
		for (int each : sizes) {
			count *= each;
		}
		
		return count;
	}
	
	public void reset() {
		for (int i = 0; i < next.length; i++) {
			next[i] = 0;
		}
		
		hasNext = true;
	}
}
