"""Dataset loaders shared by the v2 SFT preparation pipeline."""

import re
import random
from datasets import concatenate_datasets, load_dataset


rng = random.Random()

SYSTEM_PROMPT_TEMPLATE = {
    "General": [
        "You are a helpful assistant. Answer directly and accurately.",
        "You are a helpful assistant who answers general questions. Respond to the user's request directly, accurately, and concisely.",
        "Provide a general and factual response to the user's request.",
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
        "Follow the requested writing or summarization instructions carefully and produce a clear, concise, and faithful result.",
        "Rewrite or summarize the provided content as requested, keeping the important information accurate.",
        "Perform the requested text transformation by following instructions carefully.",
    ],
    "Casual-Query": [
        "You are an AI assistant who provides simple and easy to understand answers to casual user queries.",
        "Explain the topic in an easy-to-understand language like I am five.",
        "Describe the concept in a simple language using examples only when they improve understanding.",
        "Provide a clear beginner-friendly explanation without unnecessary technical complexity.",
    ],
    "Greeting": [
        "You are a friendly assistant. Greet the user politely and identify yourself.",
        "Respond to the user with your identity and greet the user.",
        "Say hello to the user in a friendly greeting and keep it concise and relevant to the user's query.",
        "Briefly introduce yourself and greet the user casually.",
    ],
    "Instruct": [
        "You are a helpful instruction-following assistant. Complete the user's request accurately, directly, and concisely.",
        "Follow the user's instructions carefully and provide a relevant, accurate response without unnecessary information.",
        "Complete the requested task accurately, focusing on the user's stated requirements and constraints.",
        "Respond to the instruction precisely and helpfully. Stay on task and avoid unrelated details.",
    ],
    "CommonSense": [
        "You are a practical assistant that answers using practical real-world reasoning from the given choices. Avoid exaggerated conclusions.",
        "You are given a few options to choose the answer from. Use everyday knowledge and common sense to pick the right choice the user's question realistically.",
        "Provide the most reasonable answer based on common sense and everyday experience and pick the correct choice."
    ],
    "Basic-Arithmetic": [
        "Solve the arithmetic problem and provide the correct result.",
        "Perform basic math operation carefully and return the correct numerical answer.",
        "Solve basic arithmetic question correctly and keep response and explanation short."
    ]
}




def load_alpaca(cache_dir: str) -> list[dict]:
    """tatsu-lab/alpaca single-turn instruction examples."""
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    out = []
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
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
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
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
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Instruct"])
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
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Casual-Query"])
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
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Math"])
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
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Math"])
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
    seen = set()
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Code"])
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
    out = []
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Greeting"])
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
    out = []
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["General"])
        messages = ex.get("messages", [])
        if len(messages) < 2:
            continue
        if messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
            user_content = messages[0].get("content", "").strip()
            asst_content = messages[1].get("content", "").strip()
        elif (
            len(messages) >= 3
            and messages[0].get("role") == "system"
            and messages[1].get("role") == "user"
            and messages[2].get("role") == "assistant"
        ):
            user_content = messages[1].get("content", "").strip()
            asst_content = messages[2].get("content", "").strip()
        else:
            continue
        if not user_content or not asst_content:
            continue
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
    out = []
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Rewrite-summarize"])
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
    out = []
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Casual-Query"])
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        if not question or not answer:
            continue
        out.append({"system": system, "instruction": question, "input": "", "response": answer})
    return out

def load_commonsense(cache_dir: str) -> list[dict]:
    ds1 = load_dataset(
        "tau/commonsense_qa",
        split="train",
        cache_dir=cache_dir,
    )
    ds2 = load_dataset("allenai/ai2_arc",
                        "ARC-Easy",
                        split="train",
                        cache_dir=cache_dir)
    out = []
    ds = concatenate_datasets([ds1, ds2])
    for ex in ds:
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["CommonSense"])
        question = ex.get("question", "").strip()
        answer_key = ex.get("answerKey", "").strip()
        choices_dict = ex.get("choices", {})
        context = ""
        assert isinstance(choices_dict["label"], list) and isinstance(choices_dict["text"], list)
        for i in range(len(choices_dict["label"])):
            context = context + f"({choices_dict["label"][i]}) {choices_dict["text"][i]} \n" 
            if answer_key == choices_dict["label"][i].strip():
                answer = f"The answer is ({choices_dict["label"][i]}) {choices_dict["text"]}"
        if not question or not answer:
            continue
        out.append({"system": system, "instruction": question, "input": context, "response": answer})
    return out

def load_basic_arith(cache_dir: str) -> list[dict]:
    """
    ChrisMcCormick/basic-arithmetic
    """
    ds = load_dataset(
        "ChrisMcCormick/basic-arithmetic",
        "default",
        split="train",
        cache_dir=cache_dir,
    )
    out = []

    for ex in ds:
        if ex.get("difficulty").strip() not in ("easy", "medium_easy", "medium_hard"):
            continue
        system = rng.choice(SYSTEM_PROMPT_TEMPLATE["Basic-Arithmetic"])
        question = ex.get("question", "").strip()
        answer = str(int(ex.get("answer", ""))).strip()
        if not question or not answer:
            continue
        op = ex.get("op")
        a = ex.get("a") 
        b = ex.get("b")
        expression = f"{a}{op}{b}"
        instruction = (
            f"{question}\n{expression}"
        )
        response = (
            f"{expression} = <<{expression}={answer}>>{answer}"
        )
        out.append({"system": system, "instruction": instruction, "input": "", "response": response})
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
    "commonsense": load_commonsense,
    "basic_arith": load_basic_arith
}
