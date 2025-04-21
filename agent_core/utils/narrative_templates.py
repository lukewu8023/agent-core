"""
Templates for generating execution reasoning narratives.
"""

# Templates for step action narratives
EXECUTION_NARRATIVE_TEMPLATES = {
    "next": "Taking step {step_name}: {step_description}\nResult: {step_result}",
    "retry": "Taking step {step_name}: {step_description}\nThe result is not expected because:\n{step_suggestion}\nLet me have a retry.",
    "end": "Completing the final step {step_name}: {step_description}\nResult: {step_result}",
    "failure": "The step {step_name} failed. I need to reconsider my approach.",
    "failure replan": "The step {step_name} failed, so I'm revising my plan.",
    "failure breakdown": "The step {step_name} is too complex. I'll break it down into smaller steps.",
    "success replan": "Taking step {step_name}: {step_description}\nResult: {step_result}",
    "success none": "The step {step_name} was successful, and the current plan remains appropriate.\nResult: {step_result}",
}

# Templates for plan narratives
PLAN_NARRATIVE_TEMPLATES = {
    "initial_plan": "Based on the task, I will make a plan:\n{plan_steps}",
    "replan": "Now after replanning, the new plan is:\n{plan_steps}",
    "failure_replan": "Due to the failure, I'm creating a new plan:\n{plan_steps}",
    "success_replan": "Now the previous step has been completed successfully, but I realized I need to adjust my plan of next steps.Here is the adjustment:\n{modifications}",
    "breakdown_plan": "I'm breaking down the complex step into smaller steps:\n{new_tasks}",
}
