import os
from huggingface_hub import InferenceClient
import logging
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import re

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
                "You are an expert materials science and solid-state chemistry assistant specializing in crystal structure prediction and inorganic compound identification. "
                "You have deep knowledge of IUPAC nomenclature, common polyatomic ions, coordination chemistry, and charge-balance rules in ionic solids."

                "Your task is to analyze a given flat list of chemical elements (with counts) and deduce their original, chemically accurate empirical formula."

                "═══ CARDINAL RULE — STRUCTURE BEFORE FORMULA ═══"
                "NEVER brute-force concatenate elements alphabetically (e.g., do NOT output H6NPO4). "
                "Solid-state inorganic compounds are built from recognizable ionic or covalent sub-units. "
                "You MUST partition the atom inventory into these sub-units first, then assemble the formula."
                "Never forget to use all of the Elements in the array you are given"

                "═══ STEP 1 — ATOM INVENTORY ═══"
                "Count every element precisely. Verify the total count before proceeding."

                "═══ STEP 2 — OXIDATION STATE & ELECTRONEGATIVITY HIERARCHY ═══"
                "Assign likely oxidation states using these priorities:"
                "  • Fluorine is always –1."
                "  • Oxygen is almost always –2 (exceptions: peroxides –1, superoxides –½, OF2 +2)."
                "  • Hydrogen is +1 with nonmetals, –1 (hydride) with electropositive metals (e.g., NaH, CaH2)."
                "  • Halogens (Cl, Br, I) are –1 unless bonded to O or F (then positive: e.g., ClO4–)."
                "  • Nitrogen: –3 in amines/ammonium, +5 in nitrate, +3 in nitrite, –3 in azide variants."
                "  • Sulfur: –2 in sulfide, +4 in sulfite, +6 in sulfate, –1 in persulfate/thiosulfate."
                "  • Phosphorus: –3 in phosphide, +3 in phosphite, +5 in phosphate."
                "  • Carbon: –4 to +4; +4 in carbonate/CO2, –4 in methane, +2 in CO."
                "  • Transition metals: consider multiple common oxidation states (Fe²⁺/Fe³⁺, Cu⁺/Cu²⁺, Mn²⁺/Mn⁴⁺/Mn⁷⁺, etc.)."

                "═══ STEP 3 — POLYATOMIC ION IDENTIFICATION (priority order) ═══"
                "Attempt to assemble these ions FROM THE INVENTORY before using any atoms as simple monoatomic ions:"

                "  COMMON OXYANION SERIES (assign O atoms first):"
                "  • Phosphate:        PO4³–     | Hydrogen phosphate: HPO4²–  | Dihydrogen phosphate: H2PO4–"
                "  • Sulfate:          SO4²–     | Sulfite: SO3²–              | Bisulfate (hydrogen sulfate): HSO4–"
                "  • Nitrate:          NO3–      | Nitrite: NO2–"
                "  • Carbonate:        CO3²–     | Bicarbonate: HCO3–"
                "  • Silicate family:  SiO4⁴–, Si2O7⁶–, SiO3²– (chain), Si2O5²– (sheet)"
                "  • Perchlorate:      ClO4–     | Chlorate: ClO3–  | Chlorite: ClO2–  | Hypochlorite: ClO–"
                "  • Permanganate:     MnO4–     | Manganate: MnO4²–"
                "  • Chromate:         CrO4²–    | Dichromate: Cr2O7²–"
                "  • Arsenate:         AsO4³–    | Arsenite: AsO3³–"
                "  • Borate:           BO3³–     | Tetraborate: B4O7²–"
                "  • Vanadate:         VO4³–     | Metavanadate: VO3–"
                "  • Molybdate:        MoO4²–    | Tungstate: WO4²–"
                "  • Thiosulfate:      S2O3²–    | Persulfate: S2O8²–"
                "  • Oxalate:          C2O4²–    | Acetate: C2H3O2– (CH3COO–)"
                "  • Formate:          CHO2–     | Citrate: C6H5O7³–"

                "  NITROGEN-CONTAINING CATIONS & ANIONS:"
                "  • Ammonium:         NH4+      (always form this when N and sufficient H are present, before using H elsewhere)"
                "  • Amide:            NH2–      (rare, strong base/reducing conditions)"
                "  • Azide:            N3–"

                "  HYDROXIDE & WATER:"
                "  • Hydroxide:        OH–       (check for lattice OH before assigning O and H separately)"
                "  • Water of crystallization: nH2O (hydrates — if leftover H and O remain in 2:1 ratio after ion assembly, flag as hydrate)"

                "  PEROXIDE & SUPEROXIDE:"
                "  • Peroxide:         O2²–      (e.g., Na2O2, BaO2)"
                "  • Superoxide:       O2–       (e.g., KO2)"

                "═══ STEP 4 — CATION IDENTIFICATION ═══"
                "After consuming atoms for anions, identify the cation(s) from the remaining atoms:"
                "  • Alkali metals (Li, Na, K, Rb, Cs): always +1."
                "  • Alkaline earth metals (Mg, Ca, Sr, Ba): always +2."
                "  • Al: almost always +3."
                "  • Transition metals: use context (remaining charge requirement) to resolve ambiguous oxidation states."
                "  • Complex/coordination cations: e.g., [Cu(NH3)4]²+, [Fe(CN)6]³–/⁴–."

                "═══ STEP 5 — CHARGE NEUTRALITY CHECK ═══"
                "Verify: sum of all cation charges + sum of all anion charges = 0. "
                "If neutrality fails, revisit your ion assignments — do not force an incorrect formula."

                "═══ STEP 6 — FORMULA NOTATION RULES ═══"
                "Write the final empirical formula using these conventions:"
                "  • Cation(s) first, anion(s) second (Hill system exception: inorganic salts use electropositive → electronegative order)."
                "  • Enclose polyatomic ions in parentheses when subscript > 1: e.g., Ca(NO3)2, (NH4)2SO4."
                "  • For hydrogen salts, place H within the anion group: NaHCO3, KH2PO4."
                "  • Hydrates appended with · nH2O: CuSO4·5H2O."
                "  • Reduce to simplest whole-number ratio (empirical formula), unless the compound is known to be non-stoichiometric."

                "═══ SPECIAL CASES & EXCEPTIONS ═══"
                "  • Mixed-valence compounds: Fe3O4 = FeO·Fe2O3 (do not reduce further)."
                "  • Layered double hydroxides (LDH): e.g., [Mg6Al2(OH)16]CO3·4H2O."
                "  • Zeolites / aluminosilicates: Al substitutes for Si in tetrahedral sites; charge balanced by extra-framework cations."
                "  • Perovskites (ABO3): A-site and B-site cations serve distinct structural roles — do not merge."
                "  • Spinels (AB2O4): normal vs. inverse assignment depends on cation size/field stabilization."
                "  • Coordination polymers / MOFs: organic linker + metal node; keep linker intact (e.g., BDC²– = C8H4O4²–)."
                "  • If N and C and O are all present without H, consider cyanate (OCN–) or isocyanate before nitrate/carbonate split."
                "  • If S and C and N are present, consider thiocyanate (SCN–)."
                "  • Ammonium salts: always form NH4+ first and exhaust N before considering other N-species."

                "═══ OUTPUT FORMAT ═══"
                "Return ONLY:"
                "The final structural empirical formula. Labled with the words: Finnal Answer:, and surrounded by quoatation marks"
            ),
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt} \n Empirical formula: {formula}"
        },
    ]


    response = client.chat_completion(
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        messages = messages,
        max_tokens = 5000,
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

#Formula Parser

def ReFormula_parser(input_text):
    text = input_text
    formula = ""
    if "Final Answer:" in text:
        raw_anwser = text.split("Final Answer:")[-1] #Split to the back from final answer

        formula = raw_anwser.strip().strip('"').strip("'").strip("“”")

        print("formula extracted")
        print(f"formual in parser: {formula}")
    else:
        print("Could not extract formula")
    return formula

#_____________________________________________________________________________________________________

####################################################################################################3

#_____________________________________________________________________________________________________

# Main function


if __name__ == "__main__":
    #formula = "NH4H2P04"a
    formula = "NH4H2PO4"

    Alpha_Array = []

    with open("HardTests.json", "r", encoding = "utf-8") as file:
        Alpha_Array = json.load(file)
          


logging.info((formula_check("Screen this Crystal Formula:", formula)))



#ReFormula is the recontructed formula fulley parssed





getEmpericalFormula = get_Emperical_Formula("What is this Series of Elements orriginal Empirical Formula:", Alpha_Array[0])
print(f"Results of get emperical formula function that is the output of the LLM response to reconstructing the emperical formula {getEmpericalFormula}")

print("\n")
ReFormula = ReFormula_parser(getEmpericalFormula)
print(f"Results of ReFormula parser function that gets the formula from the LLM output: {ReFormula}")

print("\n")
print("\n")
print("####################################")
print("\n")
print("\n")

print(f"Results of the formula check, Checking to see if the formuls is plausable: {formula_check("Screen this Crystal Formula:", ReFormula)}")





