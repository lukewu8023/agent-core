# examples/latent_reasoning_planner.py

from latent_reasoning_planner import LatentReasoningPlanner
from agent_core.evaluators import BaseEvaluator


class MockEvaluator(BaseEvaluator):
    def evaluate(self, root_task, node_task, result, background, context):
        score = 40 if "success" in result.lower() else 20
        return type("EvalResult", (), {"score": score, "details": "MockEval"})


def main():

    planner = LatentReasoningPlanner(
        model_name="gemini-1.5-flash-002",
        latent_dim=16,
        max_rec_iter=3,
        allow_human_in_the_loop=True,
        log_level="INFO",
    )

    steps_info = [
        {
            "name": "StepA",
            "description": "Gather user info and plan the approach",
            "use_tool": False,
            "info_required": ["user_age"],
        },
    ]

    my_steps = planner.plan(root_task="Do something meaningful", steps_info=steps_info)

    completed = planner.execute_plan(
        root_task="Do something meaningful",
        evaluators_enabled=True,
        evaluators=evaluators,
        background="No special background",
        tools=tools,
    )

    print("---- Execution Completed ----")
    for st in completed:
        print(f"{st.name} => result: {st.result}, is_completed: {st.is_completed}")


if __name__ == "__main__":
    main()
