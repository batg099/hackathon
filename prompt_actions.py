SYSTEM = (
    "Tu es un annotateur d'actes de dialogue dans des conversations etudiant-LLM. "
    "Tu reponds UNIQUEMENT avec du JSON valide, rien d'autre."
)

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


 r
