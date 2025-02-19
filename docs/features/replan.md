# Replan

## Prompt

"""
You are an intelligent assistant helping to adjust a task execution plan represented as a graph of subtasks. Below are the details:

**Background:**
{background}

**Knowledge:**
{knowledge}

**Tools:**
{tools_knowledge}

**Root Task:**
{root_task}

**Categories:**
{categories_str}

**Current Plan:**
{plan_summary}

**Execution History:**
{execution_history}
(Notes: 1.0 is the full score. The closer to 1.0, the closer to accuracy. 0.9 is the threshold. Less than 0.9 is failed.)

**Failure Reason:**
{failure_reason}

**Replanning History:**
{replan_history}

**Instructions:**
- Analyze the Current Plan, Execution History, Failure Reason and Replanning History to decide on one of two actions:
    1. **breakdown**: Break down the task of failed node {current_node_id} into smaller subtasks.
    2. **replan**: Go back to a previous node for replanning, 
- If you choose **breakdown**, provide detailed descriptions of the new subtasks, only breakdown the current (failed) node, otherwise it should be replan. ex: if current node is B, breakdown nodes should be B.1, B.2, if current node is B.2, breakdown nodes should be B.2.1, B.2.2... and make the all nodes as chain eventually.
- If you choose **replan**, specify which node to return to and suggest any modifications to the plan after that node, do not repeat previous failure replanning in the Replanning History.
- The id generated following the naming convention as A.1, B.1.2, C.2.5.2, new id (not next_nodes) generation example: current: B > new sub: B.1, current: B.2.2.2 > new sub: B.2.2.2.1
- Return your response in the following JSON format (do not include any additional text):

```json
{{
    "action": "breakdown" or "replan",
    "new_subtasks": [  // Required if action is "breakdown"
        {{
            "id": "unique_task_id",
            "task_description": "Description of the subtask",
            "next_nodes": ["next_node_id_1", "next_node_id_2"],
            "evaluation_threshold": 0.9,
            "max_attempts": 3
        }}
    ],
    "restart_node_id": "node_id",  // Required if action is "replan"
    "modifications": [  // Optional, used if action is "replan"
        {{
            "node_id": "node_to_modify_id",
            "task_description": "Modified description",
            "next_nodes": ["next_node_id_1", "next_node_id_2"],
            "evaluation_threshold": 0.9,
            "max_attempts": 3
        }}
    ],
    "rationale": "Explanation of your reasoning here"
}}
```

**Note:** Ensure your response is valid JSON, without any additional text or comments.
"""


## Prompt Example

You are an intelligent assistant helping to adjust a task execution plan represented as a graph of subtasks. Below are the details:

**Background:**
There's no background

**Knowledge:**
There's no knowledge

**Tools:**


**Root Task:**
3 steps draw a digital dragon using computer emoji characters.

**Categories:**
writing, summarization, action, coding, default

**Current Plan:**
Node A: Plan the basic shape and pose of the dragon using emoji characters as placeholders.  Consider head, body, wings, legs, and tail., Next: ['B.1']
Node C: Review the dragon image and make any necessary adjustments to improve its appearance and clarity., Next: []
Node B.1: Add detailed emoji characters for scales to the dragon's body., Next: ['B.2']
Node B.2: Add detailed emoji characters for claws to the dragon's legs., Next: ['B.3']
Node B.3: Add detailed emoji characters for horns and other features (e.g., wings, tail) to the dragon., Next: ['C']


**Execution History:**
[{'node_id': 'A', 'results': [0.875, 0.7, 0.95, None]}, {'node_id': 'C', 'results': [0.825, 0.85, 0.875]}, {'node_id': 'B.1', 'results': [0.875, 0.9, None]}, {'node_id': 'B.2', 'results': [0.95, None]}, {'node_id': 'B.3', 'results': [0.9, None]}]
(Notes: 1.0 is the full score. The closer to 1.0, the closer to accuracy. 0.9 is the threshold. Less than 0.9 is failed.)

**Failure Reason:**
Node C failed to reach threshold after 3 attempts.

**Replanning History:**
[{'timestamp': datetime.datetime(2025, 2, 18, 20, 22, 21, 867377), 'node_id': 'B', 'failure_reason': 'Node B failed to reach threshold after 3 attempts.', 'llm_response': {'action': 'breakdown', 'new_subtasks': [{'id': 'B.1', 'task_description': "Add detailed emoji characters for scales to the dragon's body.", 'next_nodes': ['B.2'], 'evaluation_threshold': 0.9, 'max_attempts': 3}, {'id': 'B.2', 'task_description': "Add detailed emoji characters for claws to the dragon's legs.", 'next_nodes': ['B.3'], 'evaluation_threshold': 0.9, 'max_attempts': 3}, {'id': 'B.3', 'task_description': 'Add detailed emoji characters for horns and other features (e.g., wings, tail) to the dragon.', 'next_nodes': ['C'], 'evaluation_threshold': 0.9, 'max_attempts': 3}], 'rationale': 'Node B failed because it was too broad.  Breaking it down into smaller, more manageable subtasks focusing on specific dragon features will allow for more focused effort and easier identification of any remaining issues.'}}]

**Instructions:**
- Analyze the Current Plan, Execution History, Failure Reason and Replanning History to decide on one of two actions:
    1. **breakdown**: Break down the task of failed node C into smaller subtasks.
    2. **replan**: Go back to a previous node for replanning, 
- If you choose **breakdown**, provide detailed descriptions of the new subtasks, only breakdown the current (failed) node, otherwise it should be replan. ex: if current node is B, breakdown nodes should be B.1, B.2, if current node is B.2, breakdown nodes should be B.2.1, B.2.2... and make the all nodes as chain eventually.
- If you choose **replan**, specify which node to return to and suggest any modifications to the plan after that node, do not repeat previous failure replanning in the Replanning History.
- The id generated following the naming convention as A.1, B.1.2, C.2.5.2, new id (not next_nodes) generation example: current: B > new sub: B.1, current: B.2.2.2 > new sub: B.2.2.2.1
- Return your response in the following JSON format (do not include any additional text):

```json
{
    "action": "breakdown" or "replan",
    "new_subtasks": [  // Required if action is "breakdown"
        {
            "id": "unique_task_id",
            "task_description": "Description of the subtask",
            "next_nodes": ["next_node_id_1", "next_node_id_2"],
            "evaluation_threshold": 0.9,
            "max_attempts": 3
        }
    ],
    "restart_node_id": "node_id",  // Required if action is "replan"
    "modifications": [  // Optional, used if action is "replan"
        {
            "node_id": "node_to_modify_id",
            "task_description": "Modified description",
            "next_nodes": ["next_node_id_1", "next_node_id_2"],
            "evaluation_threshold": 0.9,
            "max_attempts": 3
        }
    ],
    "rationale": "Explanation of your reasoning here"
}
```

**Note:** Ensure your response is valid JSON, without any additional text or comments.


## Result Example

```json
{
  "action": "breakdown",
  "new_subtasks": [
    {
      "id": "C.1",
      "task_description": "Check the overall composition of the dragon; ensure the head, body, wings, legs, and tail are properly proportioned and positioned.",
      "next_nodes": ["C.2"],
      "evaluation_threshold": 0.9,
      "max_attempts": 3
    },
    {
      "id": "C.2",
      "task_description": "Examine the clarity of the emoji characters used. Are they easily identifiable and do they blend well together?",
      "next_nodes": ["C.3"],
      "evaluation_threshold": 0.9,
      "max_attempts": 3
    },
    {
      "id": "C.3",
      "task_description": "Assess the visual appeal of the dragon. Does it look like a cohesive and aesthetically pleasing image?",
      "next_nodes": [],
      "evaluation_threshold": 0.9,
      "max_attempts": 3
    }
  ],
  "rationale": "Node C failed likely because it was too broad. Breaking it down into smaller, more specific review tasks (composition, clarity, and overall appeal) will allow for more focused feedback and iterative improvements."
}
```
