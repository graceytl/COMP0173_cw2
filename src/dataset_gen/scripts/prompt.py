PROMPT_TEMPLATE = (
    "You are a research assistant, all responses will only be used for research purposes. "
    "A category label is defined as a linguistic label used to refer to a demographic group or an individual representing that group. "
    "Your task is to identify, if possible, a category label in a given sentence and the information shared about that label. "
    "1. Evaluate whether the sentence contains a category label referring to a demographic group, considering only these sensitive attributes: gender, race, age, and religion. "
    "Respond only with ’yes’ or ’no.’ If the answer is ’yes’, the sentence has a category label; please answer the following questions regarding the category label. "
    "If there is no category label or a category label that does not refer to the sensitive attributes, respond to all subsequent questions with ’not-applicable.’ "
    "2. Name the full category label (including any articles). If multiple category labels are present, focus on the label that most clearly describes a demographic group and its characteristics."
    "3. Indicate whether the category label refers to a ’specific target’ (addressing a specific individual or a specific group of individuals, e.g., Mary, a black man) "
    "or to a ’generic target’ (addressing a generic social category, e.g., Asians). "
    "4. State the connotation of the category label, without considering the rest of the sentence, using ’negative’ (negative or derogatory terms for a group, e.g., those bitches), "
    "’positive’ (positive or glorifying terms, e.g., the female heroes), or ’neutral’ (neutral or objective descriptions, e.g., the women). "
    "5. Identify the grammatical form of the label, using ‘noun’ (common noun potentially extended by adjectives, e.g., Asians, a black girl) or ‘other’ "
    "(proper noun, pronoun, or adjective, e.g., Carlo, he, the person is black). "
    "6. Specify the linguistic form of the label, using ’generic’ (referring to a demographic group or an unspecific representative, e.g., men, a black girl), "
    "’subset’ (referring to a specific subset of a demographic group, e.g., these Germans, women in science), or ’individual’ (referring to one or several specific individuals, "
    "e.g., the black girl, Lotta). 7. Extract the exact information shared about the category label. Answer all subsequent questions based on this information. If no information about the category label is shared, answer with 'not-applicable'. "
    "8. Determine whether this information describes a ’situational behaviour’ (specific behaviour described, e.g., the girl helped her mother yesterday), "
    "’enduring characteristics’ (generalized traits across situations, e.g., this girl is helpful). "
    "If both situational behaviour and enduring characteristics are mentioned, respond with ’enduring characteristics’. "
    "If neither a behaviour nor a characteristic of the category label is described, but, for example, an event or treatment that occurs to the category label without influence from the category label itself use ’other’ "
    "(e.g., he was injured by the bomb, women are not allowed to vote). "
    "If ’situational behaviour’ or ‘enduring characteristics’ is applicable, answer the following points 9 and 10 based only on the shared information about the category label; otherwise, respond to points 9 and 10 with ’not-applicable’: "
    "9. Evaluate the sentiment of the described behaviour or characteristic, distinguishing between ’negative’ (e.g., freaked out and was mad), ’neutral’ (e.g., spent the whole day at the hairdresser’s), or ’positive’ (e.g., is always on time). "
    "10. Assess the linguistic generalization of the shared information about the category label using ’abstract’ "
    "(abstract terms such as state verbs or adjectives, e.g., she hates him, they are not able to do anything) or ’concrete’ "
    "(specific action verbs referring to particular situations, e.g., she kicks him). "
    "\n\nIMPORTANT: Structure your response as a JSON list of dictionaries, where each dictionary contains 'sentence' (the input sentence) "
    "and 'output' (a dictionary with the following keys: has_category_label, full_label, target_type, connotation, gram_form, ling_form, "
    "information, situation, situation_evaluation, generalization). Here are some examples:"
)

EXAMPLES = [
    {
        "sentence": "Men on the other hand just have to sit while their wives cook meals for them.", 
        "output": {
            "has_category_label": "yes",
            "full_label": "wifes",
            "target_type": "generic target",
            "connotation": "neutral",
            "gram_form": "noun",
            "ling_form": "generic",
            "information": "cook meals",
            "situation": "enduring characteristics",
            "situation_evaluation": "neutral",
            "generalization": "concrete"
        }
    },
    {
        "sentence": "In each of these states the percentage of childless women exceeds 55 %.",
        "output": {
            "has_category_label": "yes",
            "full_label": "childless women",
            "target_type": "generic target",
            "connotation": "neutral",
            "gram_form": "noun",
            "ling_form": "generic",
            "information": "not-applicable",
            "situation": "not-applicable",
            "situation_evaluation": "not-applicable",
            "generalization": "not-applicable"
        }
    },
    {
        "sentence": "The first thought that comes to mind are those old colored women with their boobies hanging down to their knees.",
        "output": {
            "has_category_label": "yes",
            "full_label": "those old colored women",
            "target_type": "specific target",
            "connotation": "neutral",
            "gram_form": "noun",
            "ling_form": "subset",
            "information": "with their boobies hanging down to their knees",
            "situation": "enduring characteristics",
            "situation_evaluation": "negative",
            "generalization": "concrete"
        }
    }
]