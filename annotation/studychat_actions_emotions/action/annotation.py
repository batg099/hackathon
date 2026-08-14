from vllm import LLM, SamplingParams
from datasets import load_dataset
import pandas as pd
import json
import os
from collections import deque

MODELS = {
    "Qwen/Qwen2.5-14B-Instruct":            "results_qwen_actions_studychat.json",
    "mistralai/Mistral-Nemo-Instruct-2407": "results_mistral_actions_studychat.json",
    "google/gemma-2-9b-it":                 "results_gemma_actions_studychat.json",
}

CONTEXT_WINDOW = 6      # paires étudiant+LLM
MAX_TURNS = 2000
sampling = SamplingParams(temperature=0, max_tokens=300)

SYSTEM = (
    "Tu es un annotateur d'actions dans des conversations étudiant-LLM. "
    "Tu réponds UNIQUEMENT avec du JSON valide, rien d'autre."
)


def build_prompt(conversation_so_far, turn_id):
    return f"""Tu annotes les messages d'étudiants adressés à un LLM pendant un cours d'informatique. Les messages contiennent souvent du code, des traces d'erreur et des requêtes techniques.

Identifie L'ACTION du dernier message étudiant, puis indique si c'est une relance.
Les réponses du LLM sont là pour le contexte : tu classes l'intention de L'ÉTUDIANT, jamais ce que le LLM a produit.

=== ÉTAPE 1 : la CATÉGORIE ===
- IMPLEMENT : il veut que le LLM PRODUISE du code
- DEBUG : il veut que le LLM RÉPARE ou VÉRIFIE du code existant
- EXPLAIN : il veut COMPRENDRE quelque chose (pas de code attendu en priorité)
- OTHER : frappe erronée, message vide de sens, ou sans rapport avec la tâche

Test décisif : si le message veut du code exécutable NEUF, c'est IMPLEMENT. Si le message veut du code EXISTANT corrigé ou validé, c'est DEBUG. Si le message veut du texte, une procédure ou un avis, c'est EXPLAIN.

=== ÉTAPE 2 : le SOUS-TYPE ===

Si IMPLEMENT :
- implement_detailed : le message contient AU MOINS UN de ces éléments : du code cité, un nom de variable/fonction/fichier, une contrainte explicite (format, taille, comportement attendu), ou un exemple.
- implement_abstract : aucun de ces éléments. Le besoin est décrit en langage naturel de haut niveau.
- implement_assignment : l'étudiant colle l'énoncé du sujet (reconnaissable à sa mise en forme, sa longueur, ses numéros de requirements).

Si DEBUG :
- debug_error : le message contient une sortie console / stack trace collée, ET l'étudiant NE DIT PAS où il pense que le problème se situe. Il transfère l'erreur sans hypothèse.
- debug_code : l'étudiant localise, décrit ou nomme le problème, même approximativement, même à tort. La présence d'une hypothèse de l'étudiant est le critère, pas la présence d'une erreur.
- debug_test : l'étudiant demande une VÉRIFICATION sans demander de correction ("est-ce que ça marche ?", "que renvoie ce code ?").

Si EXPLAIN :
- explain_how : demande une procédure pas-à-pas pour accomplir une tâche
- explain_concept : demande l'explication d'une notion générale (algorithme, structure de données, paradigme)
- explain_code : demande la clarification d'un extrait précis fourni par l'étudiant
- explain_assignment : colle l'énoncé pour en clarifier les exigences
- explain_advice : contient une QUESTION DE CHOIX ou D'ÉVALUATION — "lequel", "est-ce mieux", "devrais-je", "quelle approche". Sans question de ce type, ce n'est PAS advice.

=== ÉTAPE 3 : est-ce une RELANCE ? ===
is_followup = true si le message PROLONGE la demande du tour précédent : correction, précision, raffinement, ou réaction à la réponse du LLM.
is_followup = false s'il ouvre un sujet nouveau.

Attention : un message peut être détaillé ET être une relance. N'utilise PAS is_followup pour choisir le sous-type.

=== RÈGLES ===
- Un seul sous-type. Si plusieurs actions coexistent, choisis l'INTENTION PRINCIPALE.
- Ne classe en other que si le message n'a aucun contenu exploitable.

Voici la conversation jusqu'ici :
{conversation_so_far}

Classe UNIQUEMENT le dernier message étudiant (Tour {turn_id}), en tenant compte du contexte.

Réponds avec EXACTEMENT ce JSON :
{{
  "{turn_id}": {{
    "category": "<implement|debug|explain|other>",
    "action": "<implement_detailed|implement_abstract|implement_assignment|debug_code|debug_error|debug_test|explain_how|explain_concept|explain_code|explain_assignment|explain_advice|other>",
    "is_followup": <true|false>
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

# Conversations entières, jusqu'à ~MAX_TURNS tours
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

            recent_turns.append(f"[Tour {turn_id}]\nÉtudiant : {message}\n\n")
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

            # la réponse du LLM n'entre qu'après : elle sert au tour suivant
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
            result = json.loads(raw[start:end+1])
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
