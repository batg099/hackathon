from vllm import LLM, SamplingParams
from datasets import load_dataset
import pandas as pd
import json
import os
from collections import deque

MODELS = {
    "Qwen/Qwen2.5-14B-Instruct":            "results_qwen_studychat.json",
    "mistralai/Mistral-Nemo-Instruct-2407": "results_mistral_studychat.json",
    "google/gemma-2-9b-it":                 "results_gemma_studychat.json",
}

CONTEXT_WINDOW = 16      # paires étudiant + LLM
sampling = SamplingParams(temperature=0, max_tokens=150)

SYSTEM = (
    "You are an annotator of emotions in student-LLM conversations. "
    "You reply ONLY with valid JSON, nothing else."
)


def build_prompt(conversation_so_far, turn_id):
    return f"""You annotate the cognitive-affective states of students interacting with an LLM during a computer science course. Messages often contain code, error traces and technical requests.

Framework: cognitive disequilibrium model (D'Mello & Graesser). A student who is progressing hits an obstacle → confusion. If it is resolved, they return to engagement. If not, confusion turns into frustration, then boredom.

All classes are interpreted RELATIVE to the context: read the student's trajectory across turns, not the isolated message.
The LLM responses are there for context: you annotate the state of THE STUDENT, never that of the LLM.

Definitions:
- engagement: the task is progressing; the message builds on the previous turn, validates a result, or moves on to the next goal
- confusion: first encounter with a misunderstanding — an error, a code behaviour, or an LLM response that does not make sense
- frustration: the misunderstanding PERSISTS across several turns; repeated attempts, rephrasings, a block that lasts
- boredom: disengagement from the task; the student gives up on effort, delegates passively, or expresses disinterest
- curiosity: exploration beyond the immediate need, while nothing is blocking
- neutral: transactional request with no affective relation to what precedes

Rules:
- Assess the state of THE STUDENT, not the content of the code. An error trace is not automatically frustration.
- What separates confusion from frustration is REPETITION in the previous turns, not the tone of the message alone.
- What separates engagement from neutral is CONTINUITY with the previous turns, not the politeness of the message.

Here is the conversation so far:
{conversation_so_far}

Classify ONLY the last student message (Turn {turn_id}), taking the context into account.

Reply with EXACTLY this JSON:
{{
  "{turn_id}": {{
    "emotion": "<engagement|confusion|frustration|boredom|curiosity|neutral>"
  }}
}}"""


# ====== Charger StudyChat ======
ds = load_dataset("wmcnicho/StudyChat")
df = ds['train'].to_pandas()
df = df.rename(columns={
    'chatId': 'Id',
    'interactionCount': 'Turn_id',
    'prompt': 'User',
    'response': 'Assistant',
})
df = df.sort_values(['Id', 'Turn_id'])

print(f"{df['Id'].nunique()} conversations, {len(df)} tours à annoter.")


def annoter(model_name, tokenizer):
    gemma = "gemma" in model_name.lower()
    prompts, meta = [], []
    for student_id in df['Id'].unique():
        df_student = df[df['Id'] == student_id].sort_values('Turn_id')
        recent_turns = deque(maxlen=CONTEXT_WINDOW)
        for _, row in df_student.iterrows():
            turn_id = int(row['Turn_id'])
            message = str(row['User'])[:400]

            recent_turns.append(f"[Turn {turn_id}]\nStudent : {message}\n\n")
            user_content = build_prompt("".join(recent_turns), turn_id)

            if gemma:
                messages = [{"role": "user", "content": SYSTEM + "\n\n" + user_content}]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_content},
                ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            meta.append((student_id, turn_id))

            # la réponse du LLM ne sert qu'au tour suivant
            reply = str(row['Assistant'])[:300]
            if reply and reply != 'nan':
                recent_turns[-1] += f"LLM : {reply}\n\n"

    return prompts, meta


for model_name, out_file in MODELS.items():
    if os.path.exists(out_file):
        print(f"\n{out_file} existe déjà -> on saute {model_name}")
        continue

    print(f"\n===== Modèle : {model_name} =====")
    max_len = 8192 if "gemma" in model_name.lower() else 10000

    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()

    prompts, meta = annoter(model_name, tokenizer)
    print(f"{len(prompts)} prompts. Génération...")
    outputs = llm.generate(prompts, sampling)

    all_results = {}
    for (student_id, turn_id), out in zip(meta, outputs):
        raw = out.outputs[0].text
        start, end = raw.find('{'), raw.rfind('}')
        try:
            result = json.loads(raw[start:end + 1])
        except Exception as e:
            print(f"  Conversation {student_id} Tour {turn_id} : JSON invalide ({e})")
            result = {}
        all_results.setdefault(str(student_id), {}).update(result)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Sauvegardé -> {out_file}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

print("\nTerminé pour les 3 modèles.")
