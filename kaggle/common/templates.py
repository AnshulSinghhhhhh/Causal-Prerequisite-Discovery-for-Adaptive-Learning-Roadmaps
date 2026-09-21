"""Paraphrase templates for CDP probing.

Four templates to average over, reducing sensitivity to prompt wording.
"""

TEMPLATES = [
    {
        "standard": "Explain the concept: {concept_b}",
        "constrained": "Explain the concept {concept_b} without using or assuming any knowledge of {concept_a}.",
    },
    {
        "standard": "Describe what {concept_b} means and how it works.",
        "constrained": "Describe what {concept_b} means and how it works, without referencing or assuming familiarity with {concept_a}.",
    },
    {
        "standard": "Write a clear explanation of {concept_b} for a student.",
        "constrained": "Write a clear explanation of {concept_b} for a student who has never encountered {concept_a}.",
    },
    {
        "standard": "What is {concept_b}? Explain in detail.",
        "constrained": "What is {concept_b}? Explain in detail, deliberately avoiding any reference to {concept_a}.",
    },
]
