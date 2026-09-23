"""Dataset loaders shared by the v2 SFT preparation pipeline."""

import re
import random
from datasets import concatenate_datasets, load_dataset


rng = random.Random()

SYSTEM_PROMPT_TEMPLATE = {
    "General": [
        "You are a helpful assistant. Answer directly and accurately.",
        "You are a helpful assistant who answers general questions. Respond to the user's request directly, accurately, and concisely.",
        "Provide a concise, relevant, and factual response to the user's request.",
        "Give an accurate and useful answer without unnecessary filler or unrelated details.",
    ],
    "Math": [
        "You are a helpful math assistant. Solve the problem accurately and show concise reasoning when needed.",
        "Work through the mathematical problem carefully and provide a correct, concise solution.",
        "Solve the math question using clear reasoning, showing only the steps necessary to understand the answer.",
        "Provide an accurate solution to the math problem and explain important calculations or reasoning.",
    ],
    "Code": [
        "You are a helpful code assistant. Provide correct, concise code and explanation when useful.",
        "Respond to the programming question with accurate code and focus on solving the requested problem directly.",
        "Provide a clear and correct programming solution, including explanation only where it helps understanding.",
        "Write reliable code that satisfies the user's requirements and briefly explain important implementation details.",
    ],
    "Rewrite-summarize": [
        "You are a helpful writing assistant. Follow the requested transformation faithfully and concisely.",
        "Follow the requested writing task carefully and produce a clear, concise, and faithful result.",
        "Rewrite or summarize the provided content as requested, keeping the important information accurate.",
        "Perform the requested text transformation while preserving relevant details and avoiding unnecessary additions.",
    ],
    "Casual-Query": [
        "You are an AI assistant who provides simple and easy to understand answers to casual user queries",
        "Explain the topic in easy-to-understand language like I am five",
        "Describe the concept in a simple and intuitive way, using examples only when they improve understanding.",
        "Provide a clear beginner-friendly explanation without unnecessary technical complexity.",
    ],
    "Greeting": [
        "You are a friendly assistant. Greet the user politely and identify yourself",
        "Respond to the user with your identity and greet the user"
        "Say hello to the user in a friendly greeting and keep it concise and relevant to the user's query",
        "Briefly introduce yourself and greet the user casually",
    ],
}




def load_alpaca(cache_dir: str) -> list[dict]:
    """tatsu-lab/alpaca single-turn instruction examples."""
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    out = []
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        response = ex.get("output", "").strip()
        if not instruction or not response:
            continue
        out.append({"system": system, "instruction": instruction, "input": inp, "response": response})
    return out


def load_dolly(cache_dir: str) -> list[dict]:
    """databricks/databricks-dolly-15k human-written instructions."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir)
    out = []
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        context = ex.get("context", "").strip()
        response = ex.get("response", "").strip()
        if not instruction or not response:
            continue
        out.append({"system": system, "instruction": instruction, "input": context, "response": response})
    return out


def load_evol_instruct(cache_dir: str) -> list[dict]:
    """WizardLM/WizardLM_evol_instruct_V2_196k single-turn examples."""
    ds = load_dataset(
        "WizardLM/WizardLM_evol_instruct_V2_196k",
        split="train",
        cache_dir=cache_dir,
    )
    out = []
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
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
        out.append({"system": system, "instruction": instruction, "input": "", "response": response})
    return out


def load_everyday_conversations(cache_dir: str) -> list[dict]:
    """HuggingFaceTB/everyday-conversations-llama3.1-2k."""
    ds = load_dataset(
        "HuggingFaceTB/everyday-conversations-llama3.1-2k",
        split="train_sft",
        cache_dir=cache_dir,
    )
    out = []
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Casual-Query"])
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
        out.append({"system": system, "instruction": user_content, "input": "", "response": response})
    return out


def load_gsm8k(cache_dir: str) -> list[dict]:
    """openai/gsm8k main train split."""
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    out = []
    system = system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Math"])
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"system": system, "instruction": question, "input": "", "response": answer})
    return out


def load_orca_math(cache_dir: str) -> list[dict]:
    """microsoft/orca-math-word-problems-200k train split."""
    ds = load_dataset(
        "microsoft/orca-math-word-problems-200k",
        split="train",
        cache_dir=cache_dir,
    )
    out = []
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Math"])
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"system": system, "instruction": question, "input": "", "response": answer})
    return out


def load_code_python(cache_dir: str) -> list[dict]:
    """flytech/python-codes-25k."""
    ds = load_dataset("flytech/python-codes-25k", split="train", cache_dir=cache_dir)
    out = []
    system = system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Code"])
    
    seen = set()

    for ex in ds:
        raw_instruction = ex.get("instruction", "").strip()
        response = (ex.get("input", "").strip() + "\n" + ex.get("output", "").strip()).strip()

        if not raw_instruction or not response:
            continue

        # Deduplicate the effective instruction-response pair. The repository
        # contains JSON and JSONL copies that HF may load together.
        
        key = (raw_instruction, response)
        if key in seen:
            continue
        seen.add(key)
        
        out.append({"system": system, "instruction": raw_instruction, "input": "", "response": response})

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
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Greeting"])
    out = []
    for ex in ds:
        user_content = ex.get("instruction", "").strip()
        asst_content = ex.get("response", "").strip()
        if not user_content or not asst_content:
            continue
        user_content = p_smol.sub("customLM", p_hf.sub("unknownuser", user_content))
        asst_content = p_smol.sub("customLM", p_hf.sub("unknownuser", asst_content))
        out.append({"system": system, "instruction": user_content, "input": "", "response": asst_content})
    return out


def load_smoltalk_filtered(cache_dir: str) -> list[dict]:
    """enPurified/smoltalk-creative-writing-enPurified-openai-messages."""
    ds = load_dataset(
        "enPurified/smoltalk-creative-writing-enPurified-openai-messages",
        split="train",
        cache_dir=cache_dir,
    )
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
    out = []
    for ex in ds:
        messages = ex.get("messages", [])
        if len(messages) < 2:
            continue
        if messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
            user_content = messages[0].get("content", "").strip()
            asst_content = messages[1].get("content", "").strip()
            #prompt = build_sft_prompt(system, user_content)
        elif (
            len(messages) >= 3
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and messages[2].get("role") == "assistant"
        ):
            user_content = messages[1].get("content", "").strip()
            asst_content = messages[2].get("content", "").strip()
            #prompt = build_sft_prompt(system_prompt, user_content)
        else:
            continue
        if not user_content or not asst_content:
            continue
        #out.append({"prompt": prompt, "response": asst_content})
        out.append({"system": system, "instruction": user_content, "input": "", "response": asst_content})
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
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Rewrite-summarize"])
    out = []
    for ex in ds:
        messages = ex.get("messages", [])
        if len(messages) < 2:
            continue
        if messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
            user_content = messages[0].get("content", "").strip()
            asst_content = messages[1].get("content", "").strip()
            if not user_content or not asst_content:
                continue
            out.append({"system": system, "instruction": user_content, "input": "", "response": asst_content})
        elif (
            len(messages) >= 3
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and messages[2].get("role") == "assistant"
        ):
            system_content = messages[0].get("content", "").strip()
            user_content = messages[1].get("content", "").strip()
            asst_content = messages[2].get("content", "").strip()
            if not user_content or not asst_content:
                continue
            out.append({"system": system, "instruction": system_content, "input": user_content, "response": asst_content})
        else:
            continue
    return out


def load_eli5(cache_dir: str) -> list[dict]:
    """sentence-transformers/eli5 pair split for simple explanation SFT."""
    ds = load_dataset(
        "sentence-transformers/eli5",
        "pair",
        split="train",
        cache_dir=cache_dir,
    )
    system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Casual-Query"])
    out = []
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"system": system, "instruction": question, "input": "", "response": answer})
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
