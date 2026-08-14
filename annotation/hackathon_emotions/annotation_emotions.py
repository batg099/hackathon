# ====== Imports ======
from vllm import LLM, SamplingParams
import pandas as pd
import json
import os
from collections import deque

# ====== Modèles à tester (nom HF -> nom du fichier de sortie) ======
MODELS = {
    "Qwen/Qwen2.5-14B-Instruct":            "results_qwen.json",
    "mistralai/Mistral-Nemo-Instruct-2407": "results_mistral.json",
    "google/gemma-2-9b-it":                 "results_gemma.json",
}

CONTEXT_WINDOW = 16
sampling = SamplingParams(temperature=0, max_tokens=200)

SYSTEM = (
    "Tu es un annotateur d'émotions dans des conversations étudiant-LLM. "
    "Tu réponds UNIQUEMENT avec du JSON valide, rien d'autre."
)


def build_prompt(conversation_so_far, turn_id):
    return f"""Tu annotes les états cognitivo-affectifs d'étudiants échangeant avec un LLM pendant un hackathon. Les messages contiennent souvent du code, des traces d'erreur et des requêtes techniques.

Cadre : modèle du déséquilibre cognitif (D'Mello & Graesser). Un étudiant qui progresse rencontre un obstacle → confusion. S'il le résout, il retourne à l'engagement. Sinon, la confusion se mue en frustration, puis en boredom.

Toutes les classes s'interprètent RELATIVEMENT au contexte : lis la trajectoire de l'étudiant à travers les tours, pas le message isolé.
Les réponses du LLM sont là pour le contexte : tu annotes l'état de L'ÉTUDIANT, jamais celui du LLM.

Définitions :
- engagement : la tâche progresse ; le message s'appuie sur le tour précédent, valide un résultat, ou enchaîne sur l'objectif suivant
- confusion : première rencontre avec une incompréhension — une erreur, un comportement du code, ou une réponse du LLM qui ne fait pas sens
- frustration : l'incompréhension PERSISTE à travers plusieurs tours ; tentatives répétées, reformulations, blocage qui dure
- boredom : désengagement de la tâche ; l'étudiant abandonne l'effort, délègue passivement, ou exprime le désintérêt
- curiosite : exploration au-delà du besoin immédiat, alors que rien ne bloque
- neutral : requête transactionnelle qui n'entretient aucun rapport affectif avec ce qui précède

Règles :
- Évalue l'état de L'ÉTUDIANT, pas le contenu du code. Une trace d'erreur n'est pas automatiquement de la frustration.
- Ce qui sépare confusion de frustration est la RÉPÉTITION dans les tours précédents, pas le ton du message seul.
- Ce qui sépare engagement de neutral est la CONTINUITÉ avec les tours précédents, pas la politesse du message.

Voici la conversation jusqu'ici :
{conversation_so_far}

Classe UNIQUEMENT le dernier message étudiant (Tour {turn_id}), en tenant compte du contexte.

Réponds avec EXACTEMENT ce JSON :
{{
  "{turn_id}": {{
    "emotion": "<engagement|confusion|frustration|boredom|curiosite|neutral>"
  }}
}}"""


# ====== Charger les données une seule fois ======
df = pd.read_csv("../stage/df_turns.csv")


# ====== Fonction qui annote tout le dataset avec UN modèle ======
def annoter(model_name, tokenizer):
    # Gemma ne supporte pas le rôle "system" -> on le fusionne dans le user
    gemma = "gemma" in model_name.lower()
    prompts, meta = [], []
    for student_id in df['Id'].unique():
        df_student = df[df['Id'] == student_id].sort_values('Turn_id')
        recent_turns = deque(maxlen=CONTEXT_WINDOW)
        for _, row in df_student.iterrows():
            turn_id = int(row['Turn_id'])
            message = str(row['User'])[:400]

            recent_turns.append(f"[Tour {turn_id}]\nÉtudiant : {message}\n\n")
            conversation_so_far = "".join(recent_turns)
            user_content = build_prompt(conversation_so_far, turn_id)

            if gemma:
                messages = [
                    {"role": "user", "content": SYSTEM + "\n\n" + user_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_content},
                ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            meta.append((int(student_id), turn_id))

            # la réponse du LLM n'entre dans le contexte qu'au tour SUIVANT
            reply = str(row['Assistant'])[:300]
            if reply and reply != 'nan':
                recent_turns[-1] += f"LLM : {reply}\n\n"

    return prompts, meta


# ====== Boucle sur les 3 modèles ======
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
            result = json.loads(raw[start:end+1])
        except Exception as e:
            print(f"  Étudiant {student_id} Tour {turn_id} : JSON invalide ({e})")
            result = {}
        all_results.setdefault(student_id, {}).update(result)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Sauvegardé -> {out_file}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

print("\nTerminé pour les 3 modèles.")
