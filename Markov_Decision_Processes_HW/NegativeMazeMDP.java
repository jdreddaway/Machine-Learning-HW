import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import rl.MazeMarkovDecisionProcess;


public class NegativeMazeMDP extends MazeMarkovDecisionProcess {
	public static final char TRAP = '!';
	
	private final Map<Integer, TerminalPoint> traps;

	public NegativeMazeMDP(char[][] maze, TerminalPoint goal, Point start, List<TerminalPoint> traps, double motionFailureProbability) {
		super(maze, goal.col, goal.row, start.col, start.row, motionFailureProbability);
		
		this.traps = new HashMap<>();
		for (TerminalPoint each : traps) {
			int stateNum = stateFor(each.col, each.row);
			this.traps.put(stateNum, each);
		}
	}
	
	@Override
	public double reward(int stateNum, int action) {
		if (traps.containsKey(stateNum)) {
			return traps.get(stateNum).reward;
		} else {
			return super.reward(stateNum, action);
		}
	}
	
	@Override
	public boolean isTerminalState(int stateNum) {
		if (traps.containsKey(stateNum)) {
			return true;
		} else {
			return super.isTerminalState(stateNum);
		}
	}

	/**
	 * 
	 * @param maze maze[row][col]
	 * @param motionFailureProbability
	 * @return
	 */
	public static NegativeMazeMDP createMaze(char[][] maze, double motionFailureProbability, double trapCost, double goalReward) {
		TerminalPoint goal = null;
		List<TerminalPoint> traps = new ArrayList<>();
		Point start = null;
		
		int rowSize = maze[0].length;
		for (int row = 0; row < maze.length; row++) {
			if (rowSize != maze[row].length) {
				throw new InvalidMazeException("Row " + row + " has a different length than the first.");
			}
			
			for (int col = 0; col < rowSize; col++) {
				char currentChar = maze[row][col];
				if (currentChar == MazeMarkovDecisionProcess.GOAL) {
					TerminalPoint otherGoal = new TerminalPoint(col, row, goalReward);
					if (goal != null) {
						throw new InvalidMazeException("Multiple goals detected. One at " + goal + " and another at " + otherGoal);
					}
					
					goal = otherGoal;
				} else if (currentChar == MazeMarkovDecisionProcess.AGENT) {
					Point otherStart = new Point(col, row);
					if (goal != null) {
						throw new InvalidMazeException("Multiple starts detected. One at " + start + " and another at " + otherStart);
					}
					
					start = otherStart;
				} else if (currentChar == TRAP) {
					traps.add(new TerminalPoint(col, row, trapCost));
				}
			}
		}
		
		return new NegativeMazeMDP(maze, goal, start, traps, motionFailureProbability);
	}
}
