from src.mcts_llm.mctsr import MCTSrGPT4o, print_tree

# question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
question = """You are a VERY SMART AI called llama3 who is very good at solving puzzles. Below is a list of input and output pairs with a pattern." 
Identify the patterns."
Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs."
The pattern should be consistent to all the examples so if you apply it to the example inputs, you get the corresponding example outputs."
The result should be the output for the test input case.

Training Examples
Example 1: Input
[
[0, 1, 0],
[1, 1, 0],
[0, 1, 0],
[0, 1, 1],
[0, 1, 0],
[1, 1, 0],]

Example 1: Output
[
[0, 2, 0],
[2, 2, 0],
[0, 2, 0],
[0, 2, 2],
[0, 2, 0],
[2, 2, 0],
[0, 2, 0],
[0, 2, 2],
[0, 2, 0],]

Example 2: Input
[
[0, 1, 0],
[1, 0, 1],
[0, 1, 0],
[1, 0, 1],
[0, 1, 0],
[1, 0, 1],]

Example 2: Output
[
[0, 2, 0],
[2, 0, 2],
[0, 2, 0],
[2, 0, 2],
[0, 2, 0],
[2, 0, 2],
[0, 2, 0],
[2, 0, 2],
[0, 2, 0],]

Example 3: Input
[
[0, 1, 0],
[1, 1, 0],
[0, 1, 0],
[0, 1, 0],
[1, 1, 0],
[0, 1, 0],]

Example 3: Output
[
[0, 2, 0],
[2, 2, 0],
[0, 2, 0],
[0, 2, 0],
[2, 2, 0],
[0, 2, 0],
[0, 2, 0],
[2, 2, 0],
[0, 2, 0],]

Test
[
[1, 1, 1]
[0, 1, 0]
[0, 1, 0]
[1, 1, 1]
[0, 1, 0]
[0, 1, 0]]

Your Response:
"""

question = """You are a VERY SMART AI called llama3 who is very good at solving puzzles. Below is a list of input and output pairs with a pattern." 
Identify the patterns."
Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs."
The pattern should be consistent to all the examples so if you apply it to the example inputs, you get the corresponding example outputs."
The result should be the output for the test input case.

Training Examples
Example 1: Input
[
[1, 0, 0, 5, 0, 1, 0],
[0, 1, 0, 5, 1, 1, 1],
[1, 0, 0, 5, 0, 0, 0],]

Example 1: Output
[
[0, 0, 0],
[0, 2, 0],
[0, 0, 0],]

Example 2: Input
[
[1, 1, 0, 5, 0, 1, 0],
[0, 0, 1, 5, 1, 1, 1],
[1, 1, 0, 5, 0, 1, 0],]

Example 2: Output
[
[0, 2, 0],
[0, 0, 2],
[0, 2, 0],]

Example 3: Input
[
[0, 0, 1, 5, 0, 0, 0],
[1, 1, 0, 5, 1, 0, 1],
[0, 1, 1, 5, 1, 0, 1],]

Example 3: Output
[
[0, 0, 0],
[2, 0, 0],
[0, 0, 2],]

Test
[
[1, 0, 1, 5, 1, 0, 1]
[0, 1, 0, 5, 1, 0, 1]
[1, 0, 1, 5, 0, 1, 0]]

Your Response:
"""
mctsr = MCTSrGPT4o(
    problem = question,
    max_rollouts = 4
    )
print(mctsr)
print(mctsr.get_best_answer())
print_tree(mctsr.root)

mctsr.run()
