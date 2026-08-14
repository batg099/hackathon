# ====== Imports ======
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import pandas as pd
import json
import os
import re

# ====== Modèle à tester (on commence par UN seul) ======
# Décommente les autres quand Qwen est validé.
MODELS = {
    "Qwen/Qwen2.5-14B-Instruct":            "actions_qwen.json",
    "mistralai/Mistral-Nemo-Instruct-2407": "actions_mistral.json",
    "google/gemma-2-9b-it":                 "actions_gemma.json",
}

# Garde-fou : budget de tokens pour le prompt (marge laissée pour la sortie).
MAX_PROMPT_TOKENS = {
    "gemma": 8192 - 600,
    "default": 10000 - 600,
}

# ====== Liste fermée des labels valides (schéma DA StudyChat) ======
LABELS = {
    "write_code": "writing", "write_english": "writing", "conversion": "writing",
    "summarize": "writing", "writing_other": "writing",
    "edit_code": "editing", "edit_english": "editing", "editing_other": "editing",
    "python_library": "conceptual_questions",
    "programming_language": "conceptual_questions",
    "computer_science": "conceptual_questions",
    "programming_tools": "conceptual_questions",
    "mathematics": "conceptual_questions",
    "other_concept": "conceptual_questions",
    "assignment_clarification": "contextual_questions",
    "code_explanation": "contextual_questions",
    "interpret_output": "contextual_questions",
    "contextual_other": "contextual_questions",
    "assignment_information": "provide_context",
    "error_message": "provide_context",
    "code": "provide_context",
    "context_other": "provide_context",
    "verify_code": "verification", "verify_report": "verification",
    "verify_output": "verification", "verify_other": "verification",
    "chit_chat": "off_topic", "greeting": "off_topic",
    "gratitude": "off_topic", "offtopic_other": "off_topic",
    "misc_other": "misc",
}
VALID = set(LABELS.keys())

# ====== Guided decoding : force "specific" a etre dans la liste fermee ======
guided = GuidedDecodingParams(json={
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "broad": {"type": "string"},
            "specific": {"type": "string", "enum": sorted(VALID)},
        },
        "required": ["specific"],
    },
})
sampling = SamplingParams(temperature=0, max_tokens=200, guided_decoding=guided)

SYSTEM = (
    "Tu es un annotateur d'actes de dialogue dans des conversations etudiant-LLM. "
    "Tu reponds UNIQUEMENT avec du JSON valide, rien d'autre."
)


def truncate_head_tail(text, head, tail):
    """Garde le début ET la fin d'un texte trop long, avec un marqueur au milieu.
    Un message court (<= head+tail) est renvoyé intact, sans marqueur ni coupe."""
    text = str(text)
    if len(text) <= head + tail:
        return text
    return text[:head] + " [...] " + text[-tail:]


def build_prompt(conversation_so_far, turn_id):
    return f"""Tu annotes les ACTES DE DIALOGUE d'etudiants echangeant avec un LLM pendant un hackathon de machine learning (tache Kaggle, dataset Titanic). Les etudiants travaillent sous pression temporelle et utilisent le LLM comme assistant de code. Les messages contiennent souvent du code, des traces d'erreur et des requetes techniques.

Tu classes UNIQUEMENT le dernier message etudiant selon la FONCTION qu'il remplit dans la conversation, pas selon son sujet. Les reponses du LLM sont la pour le contexte : tu annotes le message de L'ETUDIANT, jamais celui du LLM.

Note : les messages tres longs sont abreges, le milieu etant remplace par le marqueur " [...] ". Le debut et la fin du message sont conserves.

Schema (categorie large > acte precis) :

WRITING — l'etudiant demande au LLM de PRODUIRE du code ou du texte
  write_code : generer du code nouveau
  write_english : rediger du texte (rapport, commentaire, explication)
  conversion : convertir du code/des donnees d'une forme a une autre
  summarize : resumer un texte ou une sortie
  writing_other : autre demande de production

EDITING — l'etudiant demande de MODIFIER du code ou du texte qu'il possede deja
  edit_code : modifier du code existant
  edit_english : modifier du texte existant
  editing_other : autre demande de modification

CONCEPTUAL_QUESTIONS — question de connaissance GENERALE, non liee au code courant
  python_library : sur une bibliotheque Python (pandas, sklearn...)
  programming_language : sur le langage lui-meme (syntaxe, semantique Python)
  computer_science : sur un concept CS/ML general
  programming_tools : sur un outil (git, jupyter, environnement)
  mathematics : sur un concept mathematique/statistique
  other_concept : autre question de connaissance generale

CONTEXTUAL_QUESTIONS — question liee a la tache SPECIFIQUE ou a la conversation en cours
  assignment_clarification : ce que la tache attend, comment l'interpreter
  code_explanation : demander ce que fait un morceau de code precis / pourquoi
  interpret_output : demander comment lire un resultat ou une sortie precise
  contextual_other : autre question specifique a la tache

PROVIDE_CONTEXT — l'etudiant FOURNIT de l'information, sans demande explicite
  assignment_information : colle l'enonce / les consignes de la tache
  error_message : colle une erreur console ou une trace
  code : colle du code (sans question explicite attachee)
  context_other : fournit un autre contexte

VERIFICATION — l'etudiant demande au LLM de VERIFIER ou confirmer quelque chose
  verify_code : ce code est-il correct ?
  verify_report : ce texte/rapport est-il correct ?
  verify_output : ce resultat/score est-il correct ?
  verify_other : autre demande de verification

OFF_TOPIC — sans rapport avec la tache
  chit_chat : discussion informelle
  greeting : bonjour / au revoir
  gratitude : remerciement
  offtopic_other : autre hors-sujet

MISC
  misc_other : n'entre dans aucune categorie ci-dessus

Frontieres critiques :
- CONCEPTUAL vs CONTEXTUAL : conceptual = general ("comment marche groupby ?") ; contextual = a propos de CETTE tache/code ("pourquoi mon groupby renvoie NaN ici ?").
- WRITING vs EDITING : writing = produire du neuf ; editing = changer quelque chose que l'etudiant a deja.
- PROVIDE_CONTEXT vs QUESTION : si l'etudiant colle une erreur ou du code SANS question explicite, c'est provide_context (error_message / code). Avec une question, classe la question.
- Operations git, environnement, outils -> programming_tools. Verifier un score/resultat -> verify_output.
- Si plusieurs fonctions s'appliquent, choisis la SEULE dominante.

Voici la conversation jusqu'ici :
{conversation_so_far}

Classe UNIQUEMENT le dernier message etudiant (Tour {turn_id}), en tenant compte du contexte.

Reponds avec EXACTEMENT ce JSON (le label "specific" doit etre l'un des labels exacts du schema) :
{{
  "{turn_id}": {{
    "broad": "<categorie_large>",
    "specific": "<acte_precis>"
  }}
}}"""


# ====== Charger les données une seule fois ======
df = pd.read_csv("../stage/df_turns.csv")


def _make_text(tokenizer, gemma, conversation_so_far, turn_id):
    user_content = build_prompt(conversation_so_far, turn_id)
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
    n_tok = len(tokenizer(text)["input_ids"])
    return text, n_tok


# ====== Fonction qui annote tout le dataset avec UN modèle ======
def annoter(model_name, tokenizer):
    gemma = "gemma" in model_name.lower()
    budget = MAX_PROMPT_TOKENS["gemma"] if gemma else MAX_PROMPT_TOKENS["default"]

    prompts, meta = [], []
    n_truncated = 0
    n_over_budget = 0

    for student_id in df['Id'].unique():
        df_student = df[df['Id'] == student_id].sort_values('Turn_id')
        recent_turns = []
        for _, row in df_student.iterrows():
            turn_id = int(row['Turn_id'])
            message = truncate_head_tail(row['User'], 250, 150)

            recent_turns.append(f"[Tour {turn_id}]\nÉtudiant : {message}\n\n")

            start = 0
            truncated = False
            while True:
                conversation_so_far = "".join(recent_turns[start:])
                text, n_tok = _make_text(tokenizer, gemma, conversation_so_far, turn_id)
                if n_tok <= budget:
                    break
                if start >= len(recent_turns) - 1:
                    n_over_budget += 1
                    break
                start += 1
                truncated = True
            if truncated:
                n_truncated += 1

            prompts.append(text)
            meta.append((int(student_id), turn_id))

            reply = truncate_head_tail(row['Assistant'], 200, 100)
            if reply and reply != 'nan':
                recent_turns[-1] += f"LLM : {reply}\n\n"

    print(f"  [contexte] {n_truncated} tours tronqués (contexte réduit), "
          f"dont {n_over_budget} encore trop longs seuls.")
    return prompts, meta


# ====== Parsing tolérant + validation contre la liste fermée ======
def parse_and_validate(raw, turn_id):
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start = cleaned.find('{')
    if start == -1:
        return {}
    # extrait le premier objet JSON en équilibrant les accolades, ignore la suite
    depth, end = 0, None
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    snippet = cleaned[start:end + 1] if end is not None else cleaned[start:]
    # répare les accolades non fermées
    snippet += '}' * max(0, snippet.count('{') - snippet.count('}'))
    try:
        obj = json.loads(snippet)
    except Exception:
        return {}
    inner = obj.get(str(turn_id), obj)
    if not isinstance(inner, dict):
        return {}
    spec = str(inner.get("specific", "")).strip().lower()
    if spec not in VALID:
        return {}
    return {str(turn_id): {"broad": LABELS[spec], "specific": spec}}


# ====== Boucle sur les modèles ======
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
    n_invalid = 0
    n_shown = 0
    for (student_id, turn_id), out in zip(meta, outputs):
        raw = out.outputs[0].text
        result = parse_and_validate(raw, turn_id)
        if not result:
            n_invalid += 1
            if n_shown < 5:
                n_shown += 1
                print(f"\n--- INVALIDE #{n_shown} : Étudiant {student_id} Tour {turn_id} ---")
                print(f"    len sortie = {len(raw)} chars")
                print(f"    RAW = {raw[:500]!r}")
                print("---------------------------------------------------")
        all_results.setdefault(student_id, {}).update(result)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé -> {out_file}  ({n_invalid} tours invalides sur {len(prompts)})")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

print("\nTerminé.")
