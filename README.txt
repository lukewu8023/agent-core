# PhAENix - Single LLM Agent Development Framework

PhAENix is an advanced, developer-friendly framework for building **single reliable LLM-based autonomous agents**, integrating **task planning**, **evaluation**, and **execution** into a seamless workflow. It abstracts complex aspects such as prompt engineering, multi-model coordination, and operational flow management—enabling developers to create powerful agents without being overwhelmed by technical intricacies. With a flexible design, PhAENix supports a wide range of use cases while allowing fine-grained control over each component.

- **R2D2 Native Support**: PhAENix provides seamless connectivity with **R2D2**, a company-internal gateway designed for monitoring and supervising LLM calls. This integration ensures that all external LLM communications comply with corporate governance and tracking requirements. 
- **Developer-Friendly**: By hiding most of the underlying complexity, this framework makes it straightforward for developers to specify high-level goals such as planning steps and configuring validation without worrying about convoluted implementation details.
- **Planner Variety**: The framework offers flexible planning options, including step-based planners like GenericPlanner and more complex node-graph planners like GraphPlanner, ensuring that both simple and intricate workflows are supported seamlessly.
- **Evaluation Architecture**: It includes a robust validation system that checks outputs at every step, allowing errors to be detected early. This iterative validation significantly enhances overall accuracy by ensuring that tasks are iteratively refined or dynamically adjusted.
- **Custom Configuration**: Users can adjust the framework to use different LLM models for different components—such as planning, validation, or execution—allowing precise control over the agent's behavior. Additionally, customizing prompts for each module allows for specific and tailored interactions at each stage.
- **Tool & UI Integration**: The framework supports dynamic tool use and integration with external APls. Developers can call functions, engage in real-time interactions, or retrieve live data as part of the agent's workflow, enabling interactive scenarios and adaptive behavior.
- **Other Agent Framework Integration**: PhAENix is designed to work smoothly alongside with other Agent development frameworks like **MS AutoGen** and **LangGraph**, allowing developers to combine the strengths of multiple agent frameworks in a single ecosystem. Whether you need robust orchestration with LangGraph or specialized features from MS AutoGen, you can easily plug these tools into your PhAENix-powered agent for a streamlined development experience.

In summary, this LLM agent framework offers the perfect balance between abstraction and customization, empowering developers to create powerful, intelligent agents with minimal effort while ensuring flexibility and precision in every step.

---

PhAENix is a variant and draws inspiration from the Phoenix, a mythical creature symbolizing rebirth, wisdom, and excellence across cultures.
The embedded AEN represents the core traits of an Agent: adaptability, evolution and nexus, highlighting the modern and innovative nature of the framework.

- **A**: Stands for Adaptability, highlighting the LLM Agent’s ability to dynamically adjust and perform well in various scenarios and tasks.
- **E**: Represents Evolution, emphasizes the self-optimization and iterative capabilities of the LLM Agent, symbolizing continuous transformation and improvement.
- **N**: Refers to Nexus, highlighting the LLM Agent’s role as a bridge between humans and knowledge, as well as humans and technology.

PhAENix seamlessly integrates the mythical rebirth of the Phoenix with the LLM Agent’s core strengths, showcasing its adaptability, self-evolution, and ability to empower users by automating repetitive tasks.

## Table of Contents

1. [Key Innovation Features](#key-innovation-features)
   - [Replan](#replan)
   - [Self Reflection](#self-reflection)
   - [Single Step Evaluation](#single-step-evaluation)
   - [Percise Context Control](#percise-context-control)
2. [Comparison with Other Frameworks](#comparison-with-other-frameworks)
3. [Installation](#installation)
4. [Usage and Examples](#usage-and-examples)
5. [Feedback & Usage Tracking](#feedback--usage-tracking)

## Key Innovative Features

Below are some notable innovation highlights that set PhAENix apart. You can further expand each item with additional details, examples, or images to illustrate its capabilities.

### Replan

This replan design provides a flexible approach by allowing two distinct failure-handling strategies: **breakdown**, which refines a failing task into simpler subtasks, and **replan**, which reroutes the execution path to an earlier node and modifies future tasks as needed. By dynamically choosing between these strategies whenever a node fails to meet its evaluation threshold, the system can adapt, restructure, and continue executing without having to stop or discard all progress, ultimately achieving more resilient and robust task completion.

_(Add more detailed explanations or images here.)_

### Self Reflection

By continuously measuring each node’s output against an evaluation threshold, the system pinpoints underperforming steps and decides whether to retry or move on to a more robust replan flow. This loop of “try-evaluate-improve” not only captures each failed attempt in the PlanGraph but also records the specific failure reasons and previous execution history, ensuring that the agent learns from past mistakes. This design fosters continuous refinement: it retains a local context of failures and updates the plan accordingly so the agent can self-diagnoses errors and auto recover until the task is completed successfully.

_(Add more detailed explanations or images here.)_

### Single Step Evaluation

Each step’s result undergoes a structured assessment across eight distinct criteria—covering areas like accuracy, completeness, relevance, clarity, consistency, instruction adherence, error analysis, and ethical compliance—to yield a numeric score out of 40. This granular scoring not only highlights the specific dimensions of quality (e.g., identifying if clarity is lacking), but also enables an informed accept-or-rerun decision if any criterion falls below an acceptable threshold.

_(Add more detailed explanations or images here.)_

### Percise Context Control

The planner’s handling of context is notably well-crafted, as each time a node is retried or replaced, the ContextManager selectively clears outdated entries and adds fresh details. By pruning stale or repetitive information, the model stays focused on only the most relevant partial history, avoiding confusion from repeated mistakes. This streamlined approach significantly elevates clarity and consistency in LLM responses, ensuring that each new attempt or subtask benefits from concise, targeted context rather than becoming entangled in excessive or irrelevant backstory.

_(Add more detailed explanations or images here.)_

## Comparison with Other Frameworks

Below is a report comparing PhAENix with two other notable frameworks, **autogen** and **langgraph**, across several key features:

| Feature                                 | **AutoGen** | **LangGraph** | **PhAENix**     |
| --------------------------------------- | ----------- | ------------- | --------------- |
| **Multi-Agent Collaboration Framework** | Yes         | Yes           | No              |
| **Conversational Agents Interaction**   | Yes         | No            | No              |
| **Fine-Grained Workflow Control**       | No          | Yes           | Yes             |
| **Graph-based Agentic Workflow**        | No          | Yes           | Yes             |
| **Status Management**                   | No          | Yes           | Yes             |
| **Replay & Debugging**                  | No          | Yes           | Yes             |
| **Tool Integration**                    | Yes         | Yes           | Yes             |
| **Single Step Evaluation**              | No          | No            | Yes             |
| **Dynamic Replaning**                   | No          | No            | Yes             |
| **Self Reflection & Recovery**          | No          | No            | Yes             |
| **Percise Context Management**          | No          | No            | Yes             |
| **Customizable LLM Prompts**            | No          | No            | Yes             |
| **Developer-Friendliness**              | ⭐️⭐️⭐️       | ⭐️⭐️⭐️⭐️       | ⭐️⭐️⭐️⭐️⭐️        |
| **Reliability**                         | ⭐️⭐️⭐️       | ⭐️⭐️⭐️         | ⭐️⭐️⭐️⭐️⭐️        |

## Installation

### Debug Mode

#### 1. Clone the repository

```bash
git clone https://github.com/lukewu8023/agent-core.git
```

#### 2. Download examples project or create your own project

```bash
git clone https://github.com/lukewu8023/agent-examples.git
cd agent-examples
```

#### 3. Install the project as a library

```bash
pip install -e path/to/agent-core
```

### Production Mode

#### 1. Download examples project or create your own project

```bash
git clone https://github.com/lukewu8023/agent-examples.git
cd agent-examples
```

#### 2. Install the project as a library

```bash
pip install agent-core
```

This will make the agents, planners, models, etc. available in your Python environment.

## Usage and Examples

For a comprehensive set of examples demonstrating how to use PhAENix in various scenarios, please visit our **[agent-examples repository](https://github.com/lukewu8023/agent-examples)**. There you will find **detailed examples** that show how to:

- Integrate different planners ([Example 1](#))
- Configure and customize validation loops ([Example 2](#))
- Set up multiple LLMs for specific subtasks ([Example 3](#))

### Simple Example

```python
from agent_core.agents import Agent

agent = Agent()
agent.execute("Who are you?")
```

### Advanced Example

```python
from agent_core.agents import Agent
from agent_core.planners import GraphPlanner
from agent_core.evaluators import CodingEvaluator

# Agent
agent = Agent(model_name="gemini-1.5-flash-002")

# Planner
planner = GraphPlanner(model_name="gemini-1.5-pro-002", log_level="DEBUG")
print("Default Planner Prompt: ", planner.prompt)
planner.prompt = "New planner prompt: xxx"
agent.planner = planner

# Evaluator
agent.enable_evaluators()
print("Default evaluator:", agent.evaluators)
coding_evaluator = CodingEvaluator(model_name="gemini-1.5-pro-002")
agent.add_evaluator("coding", coding_evaluator)

# Knowledge
agent.knowledge = "to draw a object you need to take 3 steps, 1) prepare tools, 2) prepare paper, 3) draw the object"

# Background
agent.background = "You are a professional artist"

# Context
context = agent.context
context.add_context(
    "role",
    f"""
    you are an digital artist, able to use computer character to draw digital picture.
    """,
)

# Execute
agent.execute("Who are you?")

# Execution Result
execution_responses = agent.execution_responses
execution_history = agent.execution_history
execution_result = agent.get_execution_result_summary()

```

## TODO List

- Dynamically manage context in single step
- knowledge and background RAG
- Introduce knowledge graph to collect sufficient information with human-in-loop
- Support LLM defined max attempt and threshold in evaluation
- Execution loop control
- Enrich llm related logs (input/output/prompt)

## Feedback & Usage Tracking

We value your feedback and would love to understand how you are using PhAENix. Here are some ways you can help us track and improve the project:

#### 1. Clone the Repository

- GitHub provides limited clone analytics, so if you’d like, please share your clone count or usage metrics by posting in GitHub issue ([Cloning Statistics](#)).

#### 2. Fork the Repository

- Use GitHub’s fork feature to create your own copy of the repo.
- This helps us track the number of active forks and gives us an idea of how many people are exploring or modifying PhAENix.

#### 3. Open a Usage Issue

- We have a dedicated Usage & Feedback Issue ([Usage Tracking](#)).
- Please drop by and let us know how you’re using PhAENix:
  - Are you simply testing or evaluating?
  - Have you deployed it in UAT or production environments?
  - Are you integrating it with other systems?

#### 4. Other Suggestions

- Stars & Watchers: Starring the repo and adding it to your watch list also gives us an indication of community interest.
- Pull Requests: Contribute features, fixes, or docs improvements. This helps us see real-world usage and encourages collaborative growth.

Your feedback is crucial for guiding the future development of PhAENix. Thank you for supporting and improving the framework!

---

Thank you for using PhAENix!  
If you have any questions or ideas for improvements, feel free to open an issue or contact the owner [Luke Wu](#).
