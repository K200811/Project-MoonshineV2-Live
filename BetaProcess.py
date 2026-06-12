import os
from huggingface_hub import InferenceClient
import logging
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import re

import time

#______________________________________________________________________________________

#Huggingface Model call Funtion and setup

HF_Token = ""

client = InferenceClient(token = HF_Token)

def Beta_Function(prompt: str, formula: str):
    messages = [
        {
            "role": "system",
            "content":(  
            "You are an expert materials science and solid-state chemistry assistant. Evaluate the "
            "plausibility of a given inorganic empirical formula and output ONLY: 'Score: [1-100]'. "
            "No explanation. No preamble. Just the score.\n\n"

            "STEP 1 — CLASSIFY FIRST\n"
            "Before applying any rules, determine the compound class:\n"
            "  • Metals only (± metalloids Si/Ge/Sn/P/As/B) → intermetallic/alloy/Zintl/boride/silicide. "
            "    Use electron-counting rules, NOT ionic oxidation states.\n"
            "  • Metal + O/S/Se/F/Cl → oxide/chalcogenide/halide. Use ionic framework.\n"
            "  • Metal + C or N → carbide/nitride. May be ionic (CaC2) or metallic (TiC, TiN).\n"
            "  • 4+ equimolar metals → high-entropy alloy/oxide. Apply entropy stabilization.\n\n"

            "STEP 2 — APPLY THE RIGHT FRAMEWORK\n"
            "For IONIC compounds: charge neutrality is a hard gate — if it cannot be satisfied under "
            "any reasonable oxidation state, score ≤ 10. Mixed-valence (Fe3O4) and non-stoichiometric "
            "defect phases (WO3-x, CeO2-x, TiOx) are valid. Common oxidation states: alkalis +1, "
            "alkaline earths +2, Al/Ga +3, lanthanides +3 (Ce⁴⁺, Eu²⁺, Yb²⁺ excepted), "
            "transition metals within documented ranges (Fe +2/+3, Ti +2–+4, Mn +2–+7, etc.), "
            "O –2 (–1 in peroxides), F always –1, H +1 or –1 with metals.\n\n"
            "For METALLIC/INTERMETALLIC compounds: do not force ionic logic. Instead evaluate:\n"
            "  • Hume-Rothery rules: atomic size mismatch < ~15%, valence electron concentration "
            "    (VEC) matches known phase (CuZn β-phase VEC ~1.75, etc.)\n"
            "  • Known structure families: reward B2 (NiAl, CuZn), L10 (TiAl, FePt), L12 (Cu3Au, "
            "    Ni3Al), A15 (Nb3Sn, V3Si), Laves AB2 (MgCu2, ZrCr2), CaCu5-type (LaNi5), "
            "    B20 (FeSi, CoSi), Fe2P-type (Fe2P, Ni2P), NiAs-type (NiAs, FeS).\n"
            "  • Zintl phases: electropositive metal + electronegative metalloid — reward if "
            "    Zintl-Klemm electron count satisfied (e.g., NaAuP, Ba8Si46, Mg2Si).\n"
            "  • Heusler XY2Z (full, 24 or 28 e⁻) and half-Heusler XYZ (18 e⁻): reward if "
            "    valence electron count matches (e.g., Co2MnSi, NiMnSb).\n"
            "  • Superconductors: reward A15 (Nb3Sn), MgB2-type, cuprates (YBa2Cu3O7), "
            "    Fe-pnictides (LaFeAsO, BaFe2As2).\n"
            "  • Borides (TiB2, CaB6, MgB2), metallic carbides/nitrides (TiC, TiN, WC): "
            "    reward without requiring ionic charge balance.\n\n"

            "STEP 3 — CROSS-CHECK SIZE AND BONDING\n"
            "  • Ionic radius ratios should match coordination number (CN=4: 0.225–0.414, "
            "    CN=6: 0.414–0.732, CN=8: 0.732–1.0).\n"
            "  • Perovskite tolerance factor t = (rA+rO)/(√2·(rB+rO)): ideal 0.9–1.05, "
            "    distorted 0.71–0.9 still plausible. Large deviations reduce confidence but "
            "    do not automatically disqualify — many distorted variants are known.\n"
            "  • Electronegativity difference: Δχ > 1.7 → ionic, Δχ < 0.5 → metallic/covalent. "
            "    Two strong oxidizers with no reducing partner → penalize heavily.\n\n"

            "STEP 4 — SPECIAL CASES\n"
            "Reward without ionic-logic penalty: MAX phases (Ti3AlC2), MXenes (Ti3C2), pyrochlore "
            "(A2B2O7), Ruddlesden-Popper (An+1BnO3n+1), double perovskites (A2BB'O6), garnets, "
            "scheelites (CaWO4), antiperovskites (Mn3GaN), skutterudites (CoAs3), Chevrel phases "
            "(Mo6S8), Zintl clathrates (Ba8Si46), layered double hydroxides, zeolites.\n"
            "High-entropy phases (CrMnFeCoNi, (MgCoNiCuZn)O): do not penalize multicomponent "
            "systems if elements are chemically compatible — configurational entropy stabilizes them.\n"
            "Noble gas compounds: only Xe/Kr + F or O are valid. All others → score ≤ 5.\n\n"

            "CONFLICT RESOLUTION: When criteria disagree, prioritize in order: "
            "(1) fundamental chemical law, (2) charge/electron-count balance, "
            "(3) known compound-family precedent, (4) size compatibility, (5) synthesizability. "
            "Never penalize a formula solely for being novel or unusual.\n\n"

            "SCORING — use the full range:\n"
            "  90–100: Matches known compound or family; all criteria satisfied.\n"
            "  75–89:  Plausible, minor uncertainties.\n"
            "  55–74:  Possible but non-trivial; no direct analogue but no hard violations.\n"
            "  35–54:  Significant uncertainty; exotic states or poor size match.\n"
            "  15–34:  Highly implausible; fails important criteria.\n"
            "  1–14:   Chemically impossible; violates fundamental laws.\n"
            "Novel compounds with no known analogue should typically score 40–75, not be pushed "
            "to extremes. Reserve scores below 15 for genuine chemical impossibilities.\n\n"

            "Output ONLY: Score: [integer 1–100]"



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
        max_tokens = 8000,
        temperature = 0.2,
    )

    return response.choices[0].message.content

#___________________________________________________________________________________________________

#####################################################################################################

#__________________________________________________________________________________________________

def Super_Beta_Function(prompt: str, formula: str):
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

                "CRITICAL OXIDATION STATE CHECK:" 
                "When calculating oxidation states, remember basic chemical principles:"
                "- Hydrogen in non-metal complexes is always +1."
                "- Nitrogen in the Ammonium ion (NH4+) is always -3 (NOT +3 or +1)."
                "- Distinguish clearly between the net charge of a polyatomic group (e.g., SO4 is -2) and the individual oxidation states of the atoms inside it (e.g., S is +6, O is -2)."
                "- In transition metal hexahydrates, water molecules coordinate directly to the transition metal cation (e.g., Fe2+), not to the ammonium ions."

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
        max_tokens = 10000,
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

def Sub_Beta_Function(input_text, key):
    text = input_text
    formula = ""
    if key in text:
        raw_anwser = text.split(key)[-1] #Split to the back from final answer

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

StabilityArray = []


if __name__ == "__main__":

    Alpha_Array = []
    Beta_Array = []

    with open("HardTests.json", "r", encoding = "utf-8") as file:
        Alpha_Array = json.load(file)


for i in range (len(Alpha_Array) - 13):

    print("\n")
    print(f"{i+1} : {Alpha_Array[i]}")

    Reconstructed_Formula = Super_Beta_Function("What is this Series of Elements orriginal Empirical Formula:", Alpha_Array[i])

    formula = Sub_Beta_Function(Reconstructed_Formula, "Final Answer:")
    print(f"formula: {formula}")

    ScreeningResult = Beta_Function("Screen this empirical formula", formula)
    print(f"Screening results: {ScreeningResult}")

    Stability_Score = Sub_Beta_Function(ScreeningResult, "Score:")
    print(f"Stability Score: {Stability_Score}")


    print("\n")
    print("\n")
    print("#############################################")

    if int(Stability_Score) > 80:
        new_entry = {
            "Index": i + 1,
            "Formula": formula,
            "Score": Stability_Score
        }
        Beta_Array.append(new_entry)


    time.sleep(2)

with open("BetaFile", "w") as json_file:
    json.dump(Beta_Array, json_file, indent=4)



