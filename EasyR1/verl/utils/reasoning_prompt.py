"""Prompt formatting shared by visible-chain and hidden-latent RL datasets."""

VISIBLE_REASONING_TEMPLATE = (
    "{question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the "
    "<answer> and </answer> tags. At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

LATENT_REASONING_TEMPLATE = (
    "{question}\n"
    "Please answer this question based on the visual content. Reason internally without outputting your analysis. "
    "Return only the final answer between the <answer> and </answer> tags.\n"
)


def format_reasoning_prompt(question: str, task_instruction: str, reasoning_mode: str) -> str:
    if reasoning_mode == "visible":
        template = VISIBLE_REASONING_TEMPLATE
    elif reasoning_mode == "latent":
        template = LATENT_REASONING_TEMPLATE
    else:
        raise ValueError("reasoning_mode must be either 'visible' or 'latent'.")
    return template.format(question=question) + task_instruction
