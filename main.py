from src.mcts_llm.mctsr import MCTSrGPT4o, print_tree, MCTSrLlama318B

base_question = """You are an AI expert at pattern recognition and puzzle-solving. Analyze the following input-output pairs to identify the transformation rules:

1. Visualize each input and output as a grid, where numbers represent different colors.
2. Pay close attention to the shape, dimensions, and color patterns in both input and output grids.
3. Identify a consistent transformation rule that applies to all example pairs.
4. The rule should be generalizable to any input of similar structure.

Your task:
1. Identify the pattern(s) that map the input to the output.
2. Formulate a clear, concise TRANSFORMATION RULE as a string.
3. Ensure your rule can be applied to any valid input to produce the correct output.

Remember:
- Be precise in your observations.
- Consider both spatial and numerical transformations.
- Your rule should work for all provided examples and be applicable to new inputs.
"""

question = base_question + """
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

"""

# 0520fde7
question = base_question + """
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

"""

# model = 'llama3.1'
model = 'gpt4o'

if model == 'gpt4o':
    mctsr = MCTSrGPT4o(
        problem=question,
        max_rollouts=20,
        max_tokens=1000
    )
elif model == 'llama3.1':
    mctsr = MCTSrLlama318B(
        problem=question,
        max_rollouts=20,
        max_tokens=1000
    )
else:
    raise ValueError(f"Model {model} not supported")

print(mctsr)
print(mctsr.get_best_answer())
print_tree(mctsr.root)

mctsr.run()
print(mctsr.get_best_answer())