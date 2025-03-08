class EvaluatorResult:

    def __init__(self, decision: str, score: int | float, suggestion: str, details):
        """
        decision: "Accept Output" or "Rerun Subtask"
        score: a numeric measure of quality
        suggestion: suggestion to improve
        details: optional extra data, e.g., breakdown of scores
        """
        self.decision = decision
        self.score = score
        self.suggestion = suggestion
        self.details = details
