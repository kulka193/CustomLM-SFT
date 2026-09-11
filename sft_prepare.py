"""Dataset loaders shared by the v2 SFT preparation pipeline."""

import re
from datasets import concatenate_datasets, load_dataset
ALPACA_WITH_INPUT = (
    "### SYSTEM:\n{system}\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_NO_INPUT = (
    "### SYSTEM:\n{system}\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_sft_prompt(system: str, instruction: str, input_text: str = "") -> str:
    system = system.strip()
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return ALPACA_WITH_INPUT.format(
            system=system,
            instruction=instruction,
            input=input_text,
        )
    return ALPACA_NO_INPUT.format(system=system, instruction=instruction)


def load_alpaca(cache_dir: str) -> list[dict]:
    """tatsu-lab/alpaca single-turn instruction examples."""
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    out = []
    system = (
        "You are a helpful, precise, and honest AI assistant. Analyze the "
        "instruction and any provided input context carefully. Deliver a direct, "
        "accurate, and completely factual response that fulfills the request "
        "without unnecessary filler."
    )
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        response = ex.get("output", "").strip()
        if not instruction or not response:
            continue
        out.append({"prompt": build_sft_prompt(system, instruction, inp), "response": response})
    return out


def load_dolly(cache_dir: str) -> list[dict]:
    """databricks/databricks-dolly-15k human-written instructions."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir)
    out = []
    system = (
        "Below is an instruction that describes a task. When provided with an "
        "input text or context, your response must be derived from it. Complete "
        "the request appropriately, truthfully, and directly."
    )
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        context = ex.get("context", "").strip()
        response = ex.get("response", "").strip()
        if not instruction or not response:
            continue
        out.append({"prompt": build_sft_prompt(system, instruction, context), "response": response})
    return out


def load_evol_instruct(cache_dir: str) -> list[dict]:
    """WizardLM/WizardLM_evol_instruct_V2_196k single-turn examples."""
    ds = load_dataset(
        "WizardLM/WizardLM_evol_instruct_V2_196k",
        split="train",
        cache_dir=cache_dir,
    )
    out = []
    system = (
        "You are a helpful AI Assistant. Below is an instruction that describes "
        "a task. Write a detailed, accurate and concise response that "
        "appropriately completes the request"
    )
    for ex in ds:
        conversations = ex.get("conversations", [])
        if len(conversations) < 2:
            continue
        human_turn = conversations[0]
        gpt_turn = conversations[1]
        if human_turn.get("from", "").lower() not in ("human", "user"):
            continue
        if gpt_turn.get("from", "").lower() not in ("gpt", "assistant"):
            continue
        instruction = human_turn.get("value", "").strip()
        response = gpt_turn.get("value", "").strip()
        if not instruction or not response:
            continue
        out.append({"prompt": build_sft_prompt(system, instruction), "response": response})
    return out


def load_everyday_conversations(cache_dir: str) -> list[dict]:
    """HuggingFaceTB/everyday-conversations-llama3.1-2k."""
    ds = load_dataset(
        "HuggingFaceTB/everyday-conversations-llama3.1-2k",
        split="train_sft",
        cache_dir=cache_dir,
    )
    out = []
    system = (
        "You are an AI assistant who provides simple and easy to understand "
        "answers to casual user queries"
    )
    for ex in ds:
        messages = ex.get("messages", [])
        idx = 2
        if len(messages) <= idx + 1:
            continue
        if messages[idx].get("role") == "user":
            user_content = messages[idx].get("content", "").strip()
            response = messages[idx + 1].get("content", "").strip()
        elif len(messages) > idx + 2 and messages[idx].get("role") == "assistant":
            user_content = messages[idx + 1].get("content", "").strip()
            response = messages[idx + 2].get("content", "").strip()
        else:
            continue
        if not user_content or not response:
            continue
        out.append({"prompt": build_sft_prompt(system, user_content), "response": response})
    return out


def load_gsm8k(cache_dir: str) -> list[dict]:
    """openai/gsm8k main train split."""
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    out = []
    system = (
        "You are an AI assistant who is an expert at solving math problems. "
        "Solve the given math word problem step by step using the given instructions"
    )
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"prompt": build_sft_prompt(system, question), "response": answer})
    return out


def load_orca_math(cache_dir: str) -> list[dict]:
    """microsoft/orca-math-word-problems-200k train split."""
    ds = load_dataset(
        "microsoft/orca-math-word-problems-200k",
        split="train",
        cache_dir=cache_dir,
    )
    out = []
    system = (
        "You are an AI assistant who is an expert at solving math problems. "
        "Solve the given math word problem step by step using the given instructions"
    )
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"prompt": build_sft_prompt(system, question), "response": answer})
    return out


def load_code_python(cache_dir: str) -> list[dict]:
    """flytech/python-codes-25k."""
    ds = load_dataset("flytech/python-codes-25k", split="train", cache_dir=cache_dir)
    out = []
    system = (
        "You are an expert at Python programming. Provide a concise and accurate "
        "solution to the following Python coding problem."
    )
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        response = (ex.get("input", "").strip() + "\n" + ex.get("output", "").strip()).strip()
        if not instruction or not response:
            continue
        out.append({"prompt": build_sft_prompt(system, instruction), "response": response})
    return out


def load_smollm_basics(cache_dir: str) -> list[dict]:
    """HuggingFaceTB/instruct-data-basics-smollm-H4."""
    ds = load_dataset(
        "HuggingFaceTB/instruct-data-basics-smollm-H4",
        split="train_sft",
        cache_dir=cache_dir,
    )
    p_hf = re.compile(r"Hugging\s*Face", re.IGNORECASE)
    p_smol = re.compile(r"SmolLM", re.IGNORECASE)
    system = (
        "You are a friendly assistant who responds with a casual yet short and "
        "simple greet. If instructed, introduce yourself and respond about your identity"
    )
    out = []
    for ex in ds:
        user_content = ex.get("instruction", "").strip()
        asst_content = ex.get("response", "").strip()
        if not user_content or not asst_content:
            continue
        user_content = p_smol.sub("customLM", p_hf.sub("unknownuser", user_content))
        asst_content = p_smol.sub("customLM", p_hf.sub("unknownuser", asst_content))
        out.append({"prompt": build_sft_prompt(system, user_content), "response": asst_content})
    return out


def load_smoltalk_filtered(cache_dir: str) -> list[dict]:
    """enPurified/smoltalk-creative-writing-enPurified-openai-messages."""
    ds = load_dataset(
        "enPurified/smoltalk-creative-writing-enPurified-openai-messages",
        split="train",
        cache_dir=cache_dir,
    )
    system = (
        "You are an instruction-following creative AI Assistant. Below is an "
        "instruction that describes a task. Write a response that appropriately "
        "completes the request"
    )
    out = []
    for ex in ds:
        messages = ex.get("messages", [])
        if len(messages) < 2:
            continue
        if messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
            user_content = messages[0].get("content", "").strip()
            asst_content = messages[1].get("content", "").strip()
            prompt = build_sft_prompt(system, user_content)
        elif (
            len(messages) >= 3
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and messages[2].get("role") == "assistant"
        ):
            user_content = messages[1].get("content", "").strip()
            asst_content = messages[2].get("content", "").strip()
            system_content = messages[0].get("content", "").strip()
            system_prompt = f"{system}\n\n{system_content}" if system_content else system
            prompt = build_sft_prompt(system_prompt, user_content)
        else:
            continue
        if not user_content or not asst_content:
            continue
        out.append({"prompt": prompt, "response": asst_content})
    return out


def load_smoltalk_summarize_rewrite(cache_dir: str) -> list[dict]:
    """HuggingFaceTB/smoltalk summarize + rewrite subsets."""
    ds_rewrite = load_dataset(
        "HuggingFaceTB/smoltalk",
        "smol-rewrite",
        split="train",
        cache_dir=cache_dir,
    )
    ds_summarize = load_dataset(
        "HuggingFaceTB/smoltalk",
        "smol-summarize",
        split="train",
        cache_dir=cache_dir,
    )
    ds = concatenate_datasets([ds_rewrite, ds_summarize])
    system = (
        "You are a helpful assistant. Answer the user's request directly, "
        "accurately, and concisely."
    )
    out = []
    for ex in ds:
        messages = ex.get("messages", [])
        if len(messages) < 2:
            continue
        if messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
            user_content = messages[0].get("content", "").strip()
            asst_content = messages[1].get("content", "").strip()
            prompt = build_sft_prompt(system, user_content)
        elif (
            len(messages) >= 3
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and messages[2].get("role") == "assistant"
        ):
            system_content = messages[0].get("content", "").strip()
            user_content = messages[1].get("content", "").strip()
            asst_content = messages[2].get("content", "").strip()
            prompt = build_sft_prompt(system, system_content, user_content)
        else:
            continue
        if not user_content or not asst_content:
            continue
        out.append({"prompt": prompt, "response": asst_content})
    return out


def load_eli5(cache_dir: str) -> list[dict]:
    """sentence-transformers/eli5 pair split for simple explanation SFT."""
    ds = load_dataset(
        "sentence-transformers/eli5",
        "pair",
        split="train",
        cache_dir=cache_dir,
    )
    system = (
        "You are an assistant that explains complex topics in simple, clear "
        "language for a curious non-expert. Keep the answer accurate, concrete, "
        "and easy to follow."
    )
    out = []
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"prompt": build_sft_prompt(system, question), "response": answer})
    return out


DATASET_LOADERS = {
    "alpaca": load_alpaca,
    "dolly": load_dolly,
    "evol_instruct": load_evol_instruct,
    "everyday_conversations": load_everyday_conversations,
    "gsm8k": load_gsm8k,
    "code_python": load_code_python,
    "smollm_basics": load_smollm_basics,
    "orca_math": load_orca_math,
    "smoltalk_10k": load_smoltalk_filtered,
    "smoltalk_summarize_rewrite": load_smoltalk_summarize_rewrite,
    "eli5": load_eli5,
}
