"""
Templates for generating execution reasoning narratives.
"""

# Templates for step action narratives
EXECUTION_NARRATIVE_TEMPLATES = {
    "next": "Taking step {step_name}: {step_description}",
    "retry": "Taking step {step_name}: {step_description}\nSomething went wrong with step {step_name}. I need to have a retry.",
    "end": "Completing the final step {step_name}: {step_description}",
    "failure": "The step '{step_name}' failed. I need to reconsider my approach.",
    "failure replan": "The step {step_name} failed, so I'm revising my plan.",
    "failure breakdown": "The step '{step_name}' is too complex. I'll break it down into smaller steps.",
    "success replan": "Taking step {step_name}: {step_description}\nNow step {step_name} completed successfully, but I realized I need to adjust my plan.",
    "success none": "The step {step_name} was successful, and the current plan remains appropriate.",
}

# Templates for plan narratives
PLAN_NARRATIVE_TEMPLATES = {
    "initial_plan": "Based on the task, I will make a plan:{plan_steps}",
    "replan": "Now after replanning, the new plan is:{plan_steps}",
    "failure_replan": "Due to the failure, I'm creating a new plan:{plan_steps}",
    "success_replan": "Here is the new plan:{plan_steps}",
    "breakdown_plan": "I'm breaking down the complex step into smaller steps:{plan_steps}",
}
