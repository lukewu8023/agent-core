"""
Templates for generating execution reasoning narratives based on step actions.
"""

# Maps action types to narrative descriptions for execution reasoning
EXECUTION_NARRATIVE_TEMPLATES = {
    "next": "Taking step {step_name}: {step_description}",
    "retry": "Something went wrong with step {step_name}. I need to have a retry.",
    "end": "Completing the final step {step_name}: {step_description}",
    "failure": "The step '{step_name}' failed. I need to reconsider my approach.",
    "failure replan": "The step {step_name} failed, so I'm revising my plan.",
    "failure breakdown": "The task '{step_name}' is too complex. I'll break it down into smaller steps.",
    "success replan": "Now step {step_name} completed successfully, after that, I realized I need to adjust my plan.",
    "success none": "The step {step_name} was successful, and the current plan remains appropriate.",
}
