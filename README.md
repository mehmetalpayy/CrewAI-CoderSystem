<h1 align="center"><strong>CrewAI Coder System</strong></h1>

## Overview

**CrewAI-CoderSystem** sets up a hierarchical CrewAI workflow with multiple specialist agents (code writing, refactor, testing, debugging, and documentation). By default, it takes a problem prompt, generates code, and returns the result.

## Task Description

The goal of this project is to automate a software development pipeline by delegating responsibilities to agents:

- Code writing
- Refactor (as needed)
- Testing
- Debugging (if errors occur)
- Documentation

## Requirements

- Python 3.9+
- An OpenAI API key
- Dependencies: `crewai`, `langchain`, `langchain_openai`, `python-dotenv`

## Repository Structure

This repository contains:

1. **codersystem.py**: Defines agents and tasks, starts the hierarchical CrewAI process, and runs a sample input.
2. **requirements.txt**: Lists required Python packages.

## Installation and Setup

1. **Clone the repository**:

```bash
git clone https://github.com/mehmetalpayy/CrewAI-CoderSystem.git
cd CrewAI-CoderSystem
```

2. **Install dependencies**:

```bash
python3 -m pip install -r requirements.txt
```

3. **Set your API key**:

Create a `.env` file and add:

```bash
OPENAI_API_KEY=your_api_key_here
```

4. **Run the project**:

```bash
python3 codersystem.py
```

## Usage

Edit the `input` string in `crew.kickoff` inside `codersystem.py` to run a different task:

```python
result = crew.kickoff(inputs={"input": "Write a Python script that implements a binary search algorithm..."})
```

## Design and Engineering Decisions

- **Hierarchical Process**: Uses `Process.hierarchical` so a manager LLM coordinates the agents.
- **Multi-Agent Workflow**: Splits responsibilities into code writing, testing, debugging, refactoring, and documentation.
- **Deterministic Output**: `temperature=0` targets consistent outputs.
- **Explicit Task Steps**: Each task has defined steps and expected output to make the flow traceable.

## Limitations and Future Work

- **Model Choice**: The default model is `gpt-4` and can be updated to newer or specialized models.
- **Testing Depth**: The testing step is generic; a real test suite could be integrated.
- **Input Handling**: The prompt is hardcoded; CLI arguments or an API layer could improve usability.

## Contributing

If you want to contribute:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.

## Contact

Questions or feedback: [mehmetcompeng@gmail.com](mailto:mehmetcompeng@gmail.com)

---

Thanks for checking out CrewAI-CoderSystem. This project is a base for experimenting with multi-agent software development workflows.
