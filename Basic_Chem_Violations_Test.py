import os
from huggingface_hub import InferenceClient
import logging
import json


#______________________________________________________________________________________

#Huggingface Model call Funtion and setup

HF_Token = ""

client = InferenceClient(token = HF_Token)

def formula_check(prompt: str, formula: str):
    messages = [
        {
            "role": "system",
            "content":(
                "You are an expert materials science and solid-state chemistry assistant." 
                    "Your task is to perform an initial plausibility screening on a given empirical formula for an inorganic crystal. Using your knowledge of general chemistry principles—such as oxidation states, charge neutrality, ionic radii, and coordination preferences—evaluate how likely this formula is to represent a stable, synthesizable crystal in the real world."
                    "Evaluate the formula and provide your response using the following structured format:"
                    "1. Likelihood Score: [A score from 1 to 100, where 1 means completely impossible/highly unstable, and 100 means an existing or highly feasible stable crystal structure.]"
                    "2. Chemical Reasoning: [A concise breakdown of why you gave this score. Mention specific factors like charge balance, expected oxidation states, or common structural frameworks.]"
                    "Do not claim definitive thermodynamic stability, as this is only an initial screening. Keep your reasoning objective, scientific, and direct."
                    "Remeber to anwser with a 1 - 100 scale likleyhood rating score"
            ),
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt} \n Empirical formula: {formula}"
        },
    ]


    response = client.chat_completion(
        model = "Qwen/Qwen2.5-7B-Instruct",
        messages = messages,
        max_tokens = 1000,
        temperature = 0.2,
    )

    return response.choices[0].message.content

#___________________________________________________________________________________________________

#####################################################################################################

#__________________________________________________________________________________________________

def get_Emperical_Formula(prompt: str, formula: str):
    messages = [
        {
            "role": "system",
            "content":(
                "You are an expert materials science and solid-state chemistry assistant specializing in crystal structure prediction."

                "Your task is to analyze a given flat list of chemical elements and deduce their original, chemically accurate empirical formula."

                "CRITICAL STRUCTURAL RULE: Do not simply smash the elements together into a brute-force formula (e.g., do NOT output H6NPO4). In solid-state inorganic chemistry, elements naturally partition into stable polyatomic groups (ions) if they are present. You must explicitly look for and group these sub-units first."
                "- Look for common cations like Ammonium (NH4+)"
                "- Look for common anions like Dihydrogen Phosphate (H2PO4-), Phosphate (PO43-), Sulfate (SO42-), etc."

                "Follow these execution steps:"
                "1. Total Atom Inventory: Total up the elements provided."
                "2. Polyatomic Ion Identification: Identify what stable complex ions can be built from this exact inventory to satisfy charge neutrality."
                "3. Formulate: Write out the proper structural empirical formula separating these sub-units (e.g., NH4H2PO4)."

                "Give me back ONLY your final structural empirical formula and a brief 2-sentence chemical reasoning."
            ),
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt} \n Empirical formula: {formula}"
        },
    ]


    response = client.chat_completion(
        model = "Qwen/Qwen2.5-7B-Instruct",
        messages = messages,
        max_tokens = 1000,
        temperature = 0.2,
    )

    return response.choices[0].message.content

#____________________________________________________________________________________________________

#######################################################################################################

#_______________________________________________________________________________________________________

# Set up Logger

logging.basicConfig(
    filename='Logs.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)

#_________________________________________________________________________________________________

##################################################################################################

#________________________________________________________________________________________________

# Main function


    


if __name__ == "__main__":
    #formula = "NH4H2P04"a
    formula = "C2H3Fe"

    Alpha_Array = []

    with open("Alpha_Arrays.json", "r", encoding = "utf-8") as file:
        Alpha_Array = json.load(file)

    #for i in range (len(Alpha_Array)):
     #   Alpha_Formula = Alpha_Array[i]
#
 #       for j in range (len(Alpha_Formula)):
  #          if j == 0:
   #             count = Alpha_Formula.count("H")
    #        if Alpha_Formula[j] != Alpha_Formula[j-1]:
     #           count = Alpha_Formula.count(Alpha_Formula[j])
                

        logging.info((formula_check("Screen this Crystal Formula:", formula)))

print(get_Emperical_Formula("What is this Series of Elements orriginal Empirical Formula:", Alpha_Array[1]))


