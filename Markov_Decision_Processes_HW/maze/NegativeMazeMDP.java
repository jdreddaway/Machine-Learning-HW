package maze;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import rl.MazeMarkovDecisionProcess;


public class NegativeMazeMDP extends MazeMarkovDecisionProcess {
	public static final char TRAP = '!';
	
	private final Map<Integer, TerminalPoint> traps;

	private final double motionFailureProbability;

	public NegativeMazeMDP(char[][] maze, TerminalPoint goal, Point start, List<TerminalPoint> traps,
			double motionFailureProbability) {
		super(maze, goal.col, goal.row, start.col, start.row, motionFailureProbability);
		this.motionFailureProbability = motionFailureProbability;
		
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
	
	public static NegativeMazeMDP createMaze(File mazeFile, double motionFailureProbability,
			double trapCost, double goalReward) throws FileNotFoundException {
		char[][] maze = loadMaze(mazeFile);
		return createMaze(maze, motionFailureProbability, trapCost, goalReward);
	}

	/**
	 * 
	 * @param maze maze[row][col]
	 * @param motionFailureProbability
	 * @param trapCost Should be a positive number. Will become negative later.
	 * @param goalReward the reward for reaching the goal
	 * @return
	 */
	public static NegativeMazeMDP createMaze(char[][] maze, double motionFailureProbability,
			double trapCost, double goalReward) {
		TerminalPoint goal = null;
		List<TerminalPoint> traps = new ArrayList<>();
		Point start = null;
		
		int rowSize = maze[0].length;
		for (int row = 0; row < maze.length; row++) {
			if (rowSize != maze[row].length) {
				throw new InvalidMazeException("Line " + (row + 1) + " has a different length than the first.");
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
					if (start != null) {
						throw new InvalidMazeException("Multiple starts detected. One at " + start + " and another at " + otherStart);
					}
					
					start = otherStart;
				} else if (currentChar == TRAP) {
					traps.add(new TerminalPoint(col, row, -trapCost));
				}
			}
		}
		
		return new NegativeMazeMDP(maze, goal, start, traps, motionFailureProbability);
	}
	
	@Override
	public double transitionProbability(int fromState, int toState, int a) {
        Motion motion = Motion.create(this, a);
        MotionProbability probabilities = motion.createMotionProbability(fromState, motionFailureProbability);
        return probabilities.getProbability(toState);
    }
	
	public boolean isMoveableLocation(int x, int y) {
		boolean xInBounds = x >= 0 && x < getWidth();
		boolean yInBounds = y >= 0 && y < getHeight();
		return xInBounds && yInBounds && !isObstacle(x, y);
	}
	
	public static char[][] loadMaze(File file) throws FileNotFoundException {
		try (Scanner scan = new Scanner(file)) {
			List<List<Character>> mazeList = new ArrayList<>();
			while(scan.hasNextLine()) {
				String line = scan.nextLine();
				List<Character> lineList = new ArrayList<>();
				for (int i = 0; i < line.length(); i++) {
					lineList.add(line.charAt(i));
				}
				mazeList.add(lineList);
			}
			
			char[][] maze = new char[mazeList.size()][];
			for (int i = 0; i < mazeList.size(); i++) {
				List<Character> line = mazeList.get(i);
				maze[i] = new char[line.size()];
				for (int j = 0; j < line.size(); j++) {
					maze[i][j] = line.get(j);
				}
			}
			
			return maze;
		}
	}
}
