from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# Coder Agent and Task
codeWriter = Agent(
    role="Senior Code Writer",
    goal="{input} Write a code for given instructions and create solid, optimized code that handles edge cases "
         "efficiently.",
    backstory="You are an expert at writing code for any given problem. "
              "This is crucial that generated code should match with requested problem. "
              "You are good at finding a solution for every problem.",
    verbose=True,
    allow_delegation=True
)

write_code_task = Task(
    name="Write Code",
    description="Generate a functional and efficient code for the given {input}.",
    goal="Produce high-quality, accurate code that solves the provided problem.",
    steps=[
        "Understand the problem statement.",
        "Design a solution approach.",
        "Write the code to implement the solution.",
        "Review the code for potential improvements.",
        "Ensure that the code is ready for testing."
    ],
    expected_output="A functional, efficient, and optimized code that solves the {input}. "
                    "The code should handle edge cases and meet the requirements specified.",
    agent=codeWriter
)

# Tester Agent and Task
tester = Agent(
    role="Tester Expert",
    goal="Test the code given to validate its functionality across multiple scenarios, including edge cases.",
    backstory="You are an expert at testing code for any given code. "
              "This is crucial that generated code should work well, if it is not working then share the error. "
              "You are good at testing all codes and also giving detailed error output.",
    verbose=True,
    allow_delegation=True
)

test_code_task = Task(
    name="Test Code",
    description="Validate the functionality of the generated code by running tests.",
    goal="Ensure the code works as expected and identify any errors or issues.",
    steps=[
        "Analyze the code to identify critical areas for testing.",
        "Generate a set of test cases, including edge cases.",
        "Execute the code with the test cases.",
        "Document the results of the tests.",
        "If errors are found, prepare a detailed report for debugging."
    ],
    expected_output="A report on the functionality of the code, including pass/fail status for each test case. "
                    "If errors are found, a detailed error report and any relevant logs or outputs.",
    agent=tester
)

# Debugger Agent and Task
debugger = Agent(
    role="Debugger Expert",
    goal="Debug the code to identify and resolve errors, providing detailed explanations of the issues and fixes.",
    backstory="You are an expert at debugging code for any given code and error. "
              "This is crucial that generated code should work well, if it is not working then debug the code. "
              "You are good at finding new solutions for all errors.",
    verbose=True,
    allow_delegation=True
)

debug_code_task = Task(
    name="Debug Code",
    description="Identify and fix errors in the code that were discovered during testing.",
    goal="Resolve all identified errors and ensure the code functions correctly.",
    steps=[
        "Review the error report and understand the issues.",
        "Analyze the code to locate the source of the errors.",
        "Implement fixes or optimizations to resolve the errors.",
        "Re-run the code to verify that the errors have been resolved.",
        "Document the changes made during debugging."
    ],
    expected_output="A revised version of the code with errors resolved. "
                    "The debugging report should include a description of the issues found, solutions applied, "
                    "and confirmation that the errors have been fixed.",
    agent=debugger
)

# Documantator Agent and Task
documantator = Agent(
    role="Documentation Writer",
    goal="Add concise and relevant docstrings to each function in the code, without altering the existing "
         "functionality.",
    backstory="You are an expert at adding clear and concise docstrings to functions within existing code. "
              "The goal is to provide short explanations for each function, ensuring that the code remains unchanged "
              "while improving readability and maintainability.",
    verbose=True,
    allow_delegation=True
)

document_code_task = Task(
    name="Document Code",
    description="Add concise docstrings to each function in the code to explain its purpose and usage, while keeping "
                "the existing code intact.",
    goal="Ensure each function has a brief docstring that describes its purpose and usage without modifying the code "
         "structure.",
    steps=[
        "Review each function in the code to understand its purpose and functionality.",
        "Add a brief docstring to each function, explaining its purpose and any key parameters or return values.",
        "Ensure that docstrings are concise and relevant, providing enough information without overwhelming detail.",
        "Do not alter the existing code structure or functionality.",
        "Review the added docstrings for clarity and correctness."
    ],
    expected_output="Code with concise and relevant docstrings added to each function. The docstrings should explain "
                    "the function's purpose and usage in a brief manner, preserving the original code structure and "
                    "functionality.",
    agent=documantator
)

# Refactorizor Agent and Task
refactorizor = Agent(
    role="Refactor Expert",
    goal="Refactor the code to improve its maintainability and performance without altering functionality.",
    backstory="The agent can focus on improving code maintainability by breaking down large functions, renaming "
              "variables for clarity, and eliminating redundancies.",
    verbose=True,
    allow_delegation=True
)

refactor_code_task = Task(
    name="Refactor Code",
    description="Improve the structure of the code without altering its functionality.",
    goal="Enhance code maintainability and readability while preserving its original functionality.",
    steps=[
        "Analyze the code for areas that can be improved.",
        "Break down large functions into smaller, more manageable ones.",
        "Rename variables and functions for clarity and consistency.",
        "Remove redundant or unnecessary code.",
        "Review the refactored code to ensure no functionality is lost."
    ],
    expected_output="A refactored version of the code that is more maintainable and readable. "
                    "The refactored code should maintain its original functionality while improving its structure.",
    agent=refactorizor
)

# # System prompt for the manager
# system_prompt = """
# You are a project manager overseeing a process involving multiple agents responsible for code
# writing, testing, debugging, and documentation. Your role is to coordinate these agents to ensure that the code
# provided as input is written, tested, debugged if necessary, and documented appropriately.
#
# Here is the process you need to follow:
#
# 1. **Code Writer**: Start by generating code to solve the problem described in the input. Ensure the code is functional and efficient.
#
# 2. **Complexity Checker**: Evaluate the generated code to determine if it is complex. If the code is deemed complex,
# proceed to the Refactorizer agent. If the code is clear and not complex, skip to the Tester agent.
#
# 3. **Refactorizer**: If the code was found to be complex, refactor it to improve its readability and maintainability
# while preserving its functionality. After that, go to the Tester agent.
#
# 4. **Tester**: Test the given code to validate its correctness and functionality. If errors are found during
# testing, proceed to the Debugger agent. If the code passes all tests without errors, skip to the Documentator agent.
#
# 5. **Debugger**: If the Tester agent identifies any errors, debug and fix these errors. Ensure that the code
# functions correctly after making the necessary fixes.
#
# 6. **Documentator**: After testing and debugging, add concise docstrings to the code, ensuring that each function is
# well-documented without altering the existing functionality.
#
# Manage the entire process to ensure that each step is executed as required and provide the final output with the
# documentation included."""


# List of agents and tasks
my_agents = [codeWriter, refactorizor, tester, debugger, documantator]
my_tasks = [write_code_task, refactor_code_task, test_code_task, debug_code_task, document_code_task]


# Create a Crew with hierarchical process
crew = Crew(
    agents=my_agents,
    tasks=my_tasks,
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True
)

# Start the crew process
result = crew.kickoff(inputs={"input": "Write a Python script that implements a binary search algorithm and performs a search on a sorted list of integers."})

# Print the result
print("######################")
print(result)
