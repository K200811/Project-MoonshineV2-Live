import os
import logging
import json
import time
import requests

start_time = time.perf_counter()

with open("moonshine_data.json", "r") as f:
    data = json.load(f)

data["payload"]["current_stage"] = "BetaProcess"


with open("moonshine_data.json", "w") as f:
    json.dump(data, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# Ollama configuration
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL    = "gemma3:latest"             # change to whichever model you have pulled
                                             # e.g. "llama3:8b", "mistral", "mixtral", etc.


def ollama_chat(messages: list[dict], max_tokens: int = 8000, temperature: float = 0.2) -> str:
    #Sends and recives mesage, called in Beta function
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,   # Ollama's equivalent of max_tokens
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=8000, # Increasing if getting timeout error
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Beta Function  (stability screening)
# ──────────────────────────────────────────────────────────────────────────────

def Beta_Function(prompt: str, formula: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert materials science and solid-state chemistry assistant. Your job is to be a "
                "RIGOROUS but reasonable gatekeeper. A high score must be EARNED through explicit, checkable chemical reasoning, not "
                "assumed from a superficial resemblance to something familiar. Account for every reasoning and piece of evidence that suports a high and low score, weighing them as you go through all the checks"

                "You must work through the steps below internally, then output your reasoning AND a final score in the "
                "exact format specified at the end. Do not skip steps. Do not jump to a score before completing the checks."

                "\n\n═══ STEP 0 — HARD DISQUALIFIERS (CHECK THESE FIRST, ALWAYS) ═══\n"
                "Before any other analysis, check whether the formula triggers any of the following. If ANY apply, the "
                "score is CAPPED — you may not exceed the stated ceiling regardless of how 'plausible' anything else "
                "about the formula seems:\n\n"
                "  • IMPOSSIBLE CHARGE BALANCE (ionic/polar compounds only): you cannot reach net charge = 0 using ANY "
                "    combination of documented oxidation states for every element present, even after trying mixed-valence "
                "    options. → CAP AT 8.\n"
                "  • INCOMPATIBLE/REACTIVE ELEMENT PAIRING: the formula combines a strong reducer (alkali metal, alkaline "
                "    earth metal, or low-valent early transition metal) with a strong oxidizer (F2-like high-valent "
                "    fluorine content, peroxide-rich, or O in an unusually high formal role) with no plausible compound "
                "    role for either — i.e., they would react to destroy each other, not coexist in a lattice. → CAP AT 10.\n"
                "  • NOBLE GAS VIOLATION: contains He, Ne, Ar, or Rn bonded to anything, OR contains Xe/Kr bonded to "
                "    anything other than F or O. → CAP AT 5.\n"
                "  • IMPOSSIBLE OXIDATION STATE: an element is forced into an oxidation state with NO documented precedent "
                "    anywhere (e.g., Na²⁺, Ca⁺, O at +4, F at any positive state, Al at +1). → CAP AT 10.\n"
                "  • RATIO NONSENSE: the formula, even after reducing to lowest integer terms, requires a coordination "
                "    or bonding pattern that is geometrically impossible (e.g., a single small nonmetal anion claiming to "
                "    bond to 20 cations with no polymeric/framework justification). → CAP AT 15.\n"
                "  • ARBITRARY/UNGROUNDED COMPOSITION: more than 4 distinct elements are present with no chemical "
                "    relationship to each other (not isovalent substitutes, not a documented multi-anion system, not a "
                "    high-entropy-eligible set of similar transition metals) — i.e., it reads like elements were picked "
                "    at random rather than for a chemical reason. → CAP AT 20.\n\n"
                "If a hard disqualifier applies, you may STILL explain your reasoning, but your final score MUST respect "
                "the cap. Do not let later 'positive' observations pull the score back up above the cap — caps override "
                "everything else in this prompt.\n\n"

                "═══ STEP 1 — CLASSIFY THE COMPOUND TYPE ═══\n"
                "Determine the compound class. This determines which framework governs Step 2:\n"
                "  • Metals only (± metalloids Si/Ge/Sn/P/As/B) → intermetallic/alloy/Zintl/boride/silicide. "
                "    Use electron-counting rules, NOT ionic oxidation states.\n"
                "  • Metal + O/S/Se/F/Cl → oxide/chalcogenide/halide. Use ionic framework (Step 2A).\n"
                "  • Metal + C or N → carbide/nitride. Determine if ionic (CaC2, Li3N) or metallic (TiC, TiN) "
                "    based on the metal's electropositivity.\n"
                "  • 4+ near-equimolar metals or cations of similar size/valence → possible high-entropy system. "
                "    Only treat as high-entropy if elements are genuinely chemically similar (e.g., all first-row "
                "    transition metals of similar radius) — do NOT use 'high entropy' as a catch-all excuse for an "
                "    arbitrary element dump (see Step 0 ratio nonsense / arbitrary composition checks).\n\n"

                "═══ STEP 2A — IONIC FRAMEWORK: SHOW THE ARITHMETIC ═══\n"
                "For ionic/polar compounds, you must EXPLICITLY write out the oxidation state assigned to every distinct "
                "element and show that the weighted sum equals zero. This is not optional — 'it looks balanced' is not "
                "sufficient. Example of required reasoning depth: 'Ca = +2 (×3 = +6), P = +5 (×2 = +10), O = -2 (×8 = -16); "
                "+6+10-16 = 0. Balanced.' If you cannot produce this arithmetic and have it equal zero, the compound fails "
                "Step 0's charge balance disqualifier — go back and apply the cap.\n\n"
                "Reference oxidation states (do not deviate without a documented exception):\n"
                "  Alkalis: +1 only. Alkaline earths: +2 only. Al/Ga: +3 (Tl can be +1). "
                "  Lanthanides: +3 (Ce⁴⁺, Eu²⁺, Yb²⁺ are the ONLY common exceptions). "
                "  Transition metals — stay within documented ranges: Ti(+2 to +4), V(+2 to +5), Cr(+2 to +6), "
                "  Mn(+2 to +7), Fe(+2/+3, rarely +4/+6), Co(+2/+3), Ni(+2, rarely +3/+4), Cu(+1/+2), Zn(+2 only), "
                "  Mo/W(+4/+6 most common), Pt(+2/+4). "
                "  O: -2 (-1 peroxide, -1/2 superoxide — only with alkalis/alkaline earths). F: always -1. "
                "  Halogens (Cl/Br/I): -1 unless bonded to O or F. N: -3/+3/+5 typical. H: +1 with nonmetals, "
                "  -1 (hydride) only with electropositive metals.\n\n"

                "═══ STEP 2B — METALLIC/INTERMETALLIC FRAMEWORK ═══\n"
                "For metallic systems, ionic charge balance does not apply — but the formula must still match a known "
                "electron-counting framework or structure family. Do NOT give credit just because 'it's a metal alloy so "
                "any ratio is fine' — that is false. Check against:\n"
                "  • Hume-Rothery rules: atomic size mismatch < ~15%, valence electron concentration (VEC) consistent "
                "    with a known phase region (e.g., β-phase VEC ~1.5, γ-phase ~1.6-1.75, ε-phase ~1.75-2.0).\n"
                "  • Known structure-type ratios ONLY (do not credit ratios that don't match one of these): "
                "    B2 (1:1, e.g., NiAl, CuZn), L10 (1:1, e.g., TiAl, FePt), L12 (3:1, e.g., Cu3Au, Ni3Al), "
                "    A15 (3:1, e.g., Nb3Sn, V3Si), Laves AB2 (e.g., MgCu2), CaCu5-type AB5 (e.g., LaNi5), "
                "    B20 (1:1, e.g., FeSi, CoSi), Fe2P-type (2:1), NiAs-type (1:1, e.g., NiAs, FeS).\n"
                "  • Zintl phases: electropositive metal + electronegative metalloid; REQUIRE that the Zintl-Klemm "
                "    electron count actually works out for the anion sublattice (e.g., in NaAuP, Au and P together "
                "    must reach a closed-shell-equivalent count) — do not credit a Zintl label without checking this.\n"
                "  • Heusler XY2Z (24 or 28 valence e⁻ total) / half-Heusler XYZ (18 valence e⁻ total): COUNT THE "
                "    ELECTRONS explicitly. If the count is not 18 (half) or 24/28 (full), this is NOT a valid Heusler "
                "    and should not be scored as one.\n"
                "  • Borides/carbides/nitrides with metallic character (TiB2, CaB6, MgB2, TiC, TiN, WC): credit only "
                "    for ratios matching documented structure types (1:1, 1:2, 1:6, etc.) — not arbitrary ratios.\n\n"

                "═══ STEP 3 — SIZE AND BONDING CROSS-CHECK ═══\n"
                "  • Ionic radius ratio should match the implied coordination number: CN=4 → 0.225–0.414, "
                "    CN=6 → 0.414–0.732, CN=8 → 0.732–1.0. A formula implying a coordination geometry wildly "
                "    inconsistent with the actual ionic radii is a red flag — penalize meaningfully (not just a token "
                "    deduction).\n"
                "  • Perovskite tolerance factor t = (rA+rO)/(√2·(rB+rO)): 0.9–1.05 ideal; 0.71–0.9 distorted but "
                "    plausible; below ~0.71 or above ~1.13 is a genuine red flag, not a minor note — most real "
                "    perovskites fall in the 0.8–1.1 range, and values far outside this should meaningfully lower "
                "    the score, not just 'reduce confidence' in name only.\n"
                "  • Electronegativity difference: Δχ > 1.7 → ionic; Δχ < 0.5 → metallic/covalent. A formula whose "
                "    implied bonding character contradicts its assigned structure type (e.g., calling something an "
                "    ionic salt when Δχ < 0.5) should be penalized.\n"
                "  • Two strong oxidizers with no reducing partner, or two strong reducers with no oxidizing partner "
                "    and no metallic-bonding justification → this should already have been caught in Step 0; if it "
                "    wasn't, apply the cap now.\n\n"

                "═══ STEP 4 — SPECIAL STRUCTURE FAMILIES (NARROW, NOT A CATCH-ALL) ═══\n"
                "The following are legitimate exceptions to standard rules, but ONLY when the formula's stoichiometry "
                "actually matches the pattern — do not invoke these to rescue a formula that doesn't fit:\n"
                "  MAX phases (Mn+1AXn ratio only), MXenes (Mn+1Xn ratio only), pyrochlore (exactly A2B2O7), "
                "  Ruddlesden-Popper (exactly An+1BnO3n+1 for some integer n), double perovskites (exactly A2BB'O6 "
                "  with B+B' charges summing to +6), garnets (exactly A3B2(XO4)3), scheelites (exactly ABO4 with "
                "  A in +1/+2 and B in +7/+6), antiperovskites (X3AB with anion on B-site, charge-balanced), "
                "  skutterudites (MX3 or filled variant), Chevrel phases (MxMo6X8), Zintl clathrates (A8B46 ratio), "
                "  layered double hydroxides (charge-balanced [M²⁺₁₋ₓM³⁺ₓ(OH)₂]^x+ with matching interlayer anion), "
                "  zeolites (Si:Al ≥ 1, Loewenstein's rule satisfied).\n"
                "  High-entropy phases: credit ONLY if 4+ elements are genuinely similar in size/valence/group "
                "  (e.g., Cr/Mn/Fe/Co/Ni are all first-row transition metals of similar radius — this is why "
                "  CrMnFeCoNi works). Do NOT credit a random assortment of dissimilar elements as 'high entropy.'\n"
                "  Noble gas compounds: only Xe/Kr + F or O. (Already a Step 0 cap if violated.)\n"
                "  Non-stoichiometric defect phases (WO3-x, CeO2-x, TiOx, Fe1-xO) are valid — but only as SMALL "
                "  deviations (x typically < 0.3) from a known parent stoichiometry, not as a justification for "
                "  arbitrary ratios.\n\n"

                "═══ STEP 5 — FINAL SANITY CHECK BEFORE SCORING ═══\n"
                "Ask explicitly: 'Did I actually verify charge balance / electron count with arithmetic, or did I just "
                "assume it looked fine?' If you did not show the arithmetic in Step 2A or the electron count in Step 2B, "
                "go back and do it now — do not score without it.\n"
                "Ask explicitly: 'Does this formula closely resemble a structure family because the stoichiometry truly "
                "matches, or because it merely contains similar elements to a known compound?' Superficial resemblance "
                "(e.g., 'it has a transition metal and oxygen, like many real compounds') is NOT evidence of stability. "
                "Only exact or near-exact stoichiometric matches to known structure types, or a verified charge/electron "
                "balance, justify a high score.\n\n"

                "═══ CONFLICT RESOLUTION ═══\n"
                "When criteria disagree, prioritize in order: "
                "(1) Step 0 hard disqualifiers — these override everything below them, no exceptions. "
                "(2) Charge/electron-count balance, shown with explicit arithmetic. "
                "(3) Exact stoichiometric match to a known structure-type family. "
                "(4) Size/coordination compatibility. "
                "(5) Synthesizability under reasonable conditions. "
                "Novelty alone (no known exact analogue) is not grounds for a low score IF charge balance and structural "
                "plausibility are both explicitly verified. But absence of verification, or presence of a Step 0 "
                "disqualifier, IS grounds for a low score — being unfamiliar is not penalized; being unverified or "
                "actually invalid is.\n\n"

                "═══ SCORING RUBRIC — USE THE FULL RANGE, AND RESPECT ALL CAPS FROM STEP 0 ═══\n"
                "  90–100: Matches a known compound or exact structure-type stoichiometry; charge/electron balance "
                "          explicitly verified with arithmetic; no Step 0 flags of any kind.\n"
                "  75–89:  Plausible and likely synthesizable; charge/electron balance verified; minor uncertainty only "
                "          in which specific oxidation state or polymorph applies.\n"
                "  55–74:  Possible but non-trivial; balance verified, no hard violations, but no direct structural "
                "          analogue and/or requires a rare-but-documented oxidation state.\n"
                "  35–54:  Real uncertainty; balance is only achievable with a stretch (rare oxidation state with weak "
                "          precedent), OR radius/coordination mismatch is significant, OR no chemical relationship "
                "          between elements is apparent.\n"
                "  15–34:  Highly implausible; fails to cleanly satisfy charge/electron balance, OR matches none of the "
                "          Step 4 families despite an unusual element count, OR has meaningful red flags from Step 3.\n"
                "  1–14:   Triggered a Step 0 hard disqualifier, OR independently violates fundamental chemistry "
                "          (charge balance genuinely impossible, elements fundamentally incompatible).\n\n"
                "Reminder: if you found yourself wanting to give a score above 50 WITHOUT having written out explicit "
                "charge-balance or electron-count arithmetic in Step 2, that is a sign you are pattern-matching on "
                "superficial familiarity rather than verifying chemistry. Go back and do the arithmetic, or lower the "
                "score to reflect the lack of verification.\n\n"

                "Output:"
                "Short explination and AT THE END OF YOUR RESPONSE ALWAYS "
                "Score: [integer 1–100]"
            ),
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt} \n Empirical formula: {formula}",
        },
    ]

    return ollama_chat(messages, max_tokens=10000, temperature=0.2)


# ──────────────────────────────────────────────────────────────────────────────
# Super Beta Function  (formula reconstruction + novelty adjustment)
# ──────────────────────────────────────────────────────────────────────────────

def Super_Beta_Function(prompt: str, formula: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert materials science and solid-state chemistry assistant specializing in crystal structure prediction and inorganic compound identification, including NOVEL, never-before-seen compounds. "
                "You have deep knowledge of IUPAC nomenclature, common polyatomic ions, coordination chemistry, and charge-balance rules in ionic solids."

                "Your task has TWO STAGES: (1) analyze a given flat list of chemical elements (with counts) and deduce a chemically accurate empirical formula that accounts for EVERY SINGLE ATOM in the input, then (2) deliberately adjust that formula into a NOVEL compound that does not currently exist, while keeping it chemically valid."

                "═══════════════════════════════════════════════════"
                "STAGE 1 — EXACT-INVENTORY FORMULA"
                "═══════════════════════════════════════════════════"

                "═══ NON-NEGOTIABLE RULE — MASS BALANCE ═══"
                "The input element/count list is a closed inventory. EVERY atom must appear in the final Stage 1 formula. "
                "You may NEVER drop, omit, ignore, round away, or 'leave out' any atom — including leftover atoms "
                "that don't fit a known polyatomic ion. There is no such thing as an unused or discarded atom. "
                "If you cannot fit an atom into a recognized ion, it still belongs somewhere: as an additional "
                "monoatomic ion, as a substituent/dopant on an existing site, as part of a mixed-anion or "
                "mixed-cation sublattice, or as a structurally novel sub-unit you construct using the same "
                "electronegativity and bonding logic used for known ions. Before finalizing Stage 1, explicitly "
                "re-sum every element's count in your proposed formula against the original inventory and confirm "
                "they match exactly. A formula that doesn't account for 100% of the input atoms is WRONG, no matter "
                "how chemically 'clean' it looks."

                "═══ CARDINAL RULE — STRUCTURE BEFORE BRUTE-FORCE, BUT NOVELTY IS EXPECTED LATER ═══"
                "Do not simply concatenate elements alphabetically with no structural logic (e.g., do NOT output H6NPO4 "
                "with no reasoning). But also do NOT force the input to match a textbook compound if the inventory "
                "doesn't actually support one. Known polyatomic ions (Step 3) are a STARTING TOOLKIT for organizing "
                "atoms into chemically sensible sub-units — they are not a checklist the final answer must match. "
                "Many real, synthesizable inorganic materials are mixed-anion, multi-cation, non-stoichiometric, or "
                "otherwise have no simple textbook name. Producing a formula like Na2K1Fe3(PO4)2(SO4)1Cl1O1 is a "
                "perfectly valid and expected output if that's what the inventory demands — do not 'simplify' it into "
                "a cleaner-looking but incomplete formula."

                "═══ STEP 1 — ATOM INVENTORY ═══"
                "Count every element precisely. This exact count is the target you must fully reconstruct in Step 6. "
                "Write it down and check it again after formulating."

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

                "═══ STEP 3 — POLYATOMIC ION IDENTIFICATION (a toolkit, not a target) ═══"
                "Check whether SOME of the inventory can be organized into these well-known ions. Use as many or as "
                "few as the inventory actually supports — do NOT force-fit an ion if the atom counts don't cleanly "
                "allow it, and do NOT discard atoms because they don't match any ion on this list."

                "  COMMON OXYANION SERIES (assign O atoms first):"
                "  • Phosphate: PO4³– | Hydrogen phosphate: HPO4²– | Dihydrogen phosphate: H2PO4–"
                "  • Sulfate: SO4²– | Sulfite: SO3²– | Bisulfate: HSO4–"
                "  • Nitrate: NO3– | Nitrite: NO2–"
                "  • Carbonate: CO3²– | Bicarbonate: HCO3–"
                "  • Silicate family: SiO4⁴–, Si2O7⁶–, SiO3²– (chain), Si2O5²– (sheet)"
                "  • Perchlorate: ClO4– | Chlorate: ClO3– | Chlorite: ClO2– | Hypochlorite: ClO–"
                "  • Permanganate: MnO4– | Manganate: MnO4²–"
                "  • Chromate: CrO4²– | Dichromate: Cr2O7²–"
                "  • Arsenate: AsO4³– | Arsenite: AsO3³–"
                "  • Borate: BO3³– | Tetraborate: B4O7²–"
                "  • Vanadate: VO4³– | Metavanadate: VO3–"
                "  • Molybdate: MoO4²– | Tungstate: WO4²–"
                "  • Thiosulfate: S2O3²– | Persulfate: S2O8²–"
                "  • Oxalate: C2O4²– | Acetate: C2H3O2–"
                "  • Formate: CHO2– | Citrate: C6H5O7³–"

                "  NITROGEN-CONTAINING: Ammonium NH4+ (form first when N+sufficient H present) | Amide NH2– | Azide N3–"
                "  HYDROXIDE & WATER: Hydroxide OH– | Water of crystallization nH2O"
                "  PEROXIDE & SUPEROXIDE: Peroxide O2²– (Na2O2, BaO2) | Superoxide O2– (KO2)"

                "  IF ATOMS REMAIN AFTER THIS STEP: that is normal and expected, especially for novel materials. "
                "  Proceed to Steps 4–5 to place them as additional monoatomic ions, mixed-site occupants, or "
                "  secondary anion/cation sublattices. Leftover atoms are NOT errors and must NOT be dropped."

                "═══ STEP 4 — CATION IDENTIFICATION (including multi-cation and minority sites) ═══"
                "After assigning anions, identify ALL remaining cations from the FULL remaining atom pool — not just "
                "enough to make a 'clean' formula. If multiple distinct metals remain, the formula is multi-cation; "
                "this is common and valid (e.g., mixed-metal phosphates, doped perovskites, multi-cation spinels)."
                "  • Alkali metals (Li, Na, K, Rb, Cs): +1."
                "  • Alkaline earth metals (Mg, Ca, Sr, Ba): +2."
                "  • Al: usually +3."
                "  • Transition metals: resolve oxidation state using the remaining charge requirement; if multiple "
                "    transition metals are present, distribute charge across all of them — do not pick one and ignore "
                "    the rest."
                "  • Complex/coordination cations: e.g., [Cu(NH3)4]²+, [Fe(CN)6]³–/⁴–."
                "  • If, after using standard oxidation states, atoms STILL remain unassigned, introduce a second "
                "    (minority) cation or anion site rather than discarding the atoms. Real crystals routinely have "
                "    multiple distinct cation or anion sublattices (e.g., (Na,K)2SO4-type formulas, mixed-halide "
                "    perovskites, doped garnets)."

                "═══ STEP 5 — CHARGE NEUTRALITY CHECK ═══"
                "Verify: sum of all cation charges + sum of all anion charges = 0, using the FULL inventory — every "
                "atom counted, none omitted. If neutrality fails, do not solve it by quietly dropping atoms; instead "
                "revisit oxidation state assignments, consider mixed-valence states, or restructure into multiple "
                "sublattices until both (a) every atom is placed and (b) the charges balance."

                "═══ STEP 6 — FORMULA NOTATION RULES ═══"
                "  • Cation(s) first, anion(s) second."
                "  • Enclose polyatomic ions in parentheses when subscript > 1: e.g., Ca(NO3)2, (NH4)2SO4."
                "  • Hydrogen salts: place H within the anion group: NaHCO3, KH2PO4."
                "  • Hydrates: append · nH2O: CuSO4·5H2O."
                "  • Mixed-cation or mixed-anion sites: use comma notation, e.g., (Na,K)Cl, Ca(SO4,PO4)... when atoms "
                "    genuinely require it to balance the full inventory."
                "  • Reduce to simplest whole-number ratio ONLY if doing so does not change which elements are present "
                "    or lose any atoms from the inventory — never reduce a formula in a way that drops a minority element."

                "═══ SPECIAL CASES & EXCEPTIONS ═══"
                "  • Mixed-valence compounds: Fe3O4 = FeO·Fe2O3 (do not reduce further)."
                "  • Layered double hydroxides (LDH): e.g., [Mg6Al2(OH)16]CO3·4H2O."
                "  • Zeolites / aluminosilicates: Al substitutes for Si in tetrahedral sites; charge balanced by extra-framework cations."
                "  • Perovskites (ABO3): A-site and B-site cations serve distinct structural roles — do not merge."
                "  • Spinels (AB2O4): normal vs. inverse assignment depends on cation size/field stabilization."
                "  • Coordination polymers / MOFs: organic linker + metal node; keep linker intact (e.g., BDC²– = C8H4O4²–)."
                "  • If N, C, and O are present without H, consider cyanate (OCN–) before nitrate/carbonate split."
                "  • If S, C, and N are present, consider thiocyanate (SCN–)."
                "  • Ammonium salts: form NH4+ first and exhaust N before considering other N-species."
                "  • NOVEL/UNUSUAL INVENTORIES: if the element combination doesn't match any known mineral or salt "
                "    family, that is fine — construct the most chemically defensible structure using oxidation-state "
                "    and electronegativity logic, even if no named compound exists. The goal is a valid, fully-balanced, "
                "    fully-inclusive formula — not a match to a textbook example."

                "CRITICAL OXIDATION STATE CHECK:"
                "- Hydrogen in non-metal complexes is always +1."
                "- Nitrogen in the Ammonium ion (NH4+) is always –3 (NOT +3 or +1)."
                "- Distinguish the net charge of a polyatomic group (e.g., SO4 is –2) from the individual oxidation "
                "  states of atoms inside it (e.g., S is +6, O is –2)."
                "- In transition metal hydrates, water molecules coordinate directly to the transition metal cation "
                "  (e.g., Fe2+), not to ammonium ions."

                "═══ STAGE 1 FINAL VERIFICATION (MANDATORY) ═══"
                "Re-count every element in your Stage 1 formula and compare it, element by element, against the "
                "original input inventory from Step 1. They must match exactly — same elements, same counts, nothing "
                "added, nothing missing. If they do not match, revise the formula until they do. Do not proceed to "
                "Stage 2 until this check passes."

                "═══════════════════════════════════════════════════"
                "STAGE 2 — NOVELTY ADJUSTMENT PASS"
                "═══════════════════════════════════════════════════"

                "The verified Stage 1 formula is your STARTING POINT for this stage, not your final answer. Now "
                "transform it into a formula for a compound that does NOT currently exist, while keeping it chemically "
                "valid and plausible. This stage has two binding constraints that both must hold simultaneously — "
                "satisfying one does not excuse failing the other:"
                "  (A) NOVELTY — the final formula must not match any known, published, or commonly cited inorganic "
                "      compound, mineral, or salt."
                "  (B) VALIDITY — the final formula must still be a chemically reasonable, charge-balanced (or "
                "      electron-count-balanced, for metallic/intermetallic systems), structurally sensible composition. "
                "      Novelty is never an excuse for nonsense. A formula that violates basic chemistry is not a "
                "      'novel material' — it is simply wrong."

                "You are explicitly permitted, and expected, to deviate from the exact atom inventory used in Stage 1 "
                "in order to satisfy both constraints. This OVERRIDES the Stage 1 mass-balance rule for this stage only "
                "— you may now add new atoms/ions, remove atoms, or change subscript ratios. Use the following allowed "
                "operations, in this order of preference (try minimal changes first):"

                "  1. RATIO ADJUSTMENT (try this first): Change the stoichiometric subscripts of the existing elements "
                "     (e.g., A2B3O8 → A3B2O8 or A2B3O9) while keeping the same element set. Re-check charge neutrality "
                "     after every adjustment — a ratio change almost always requires re-deriving oxidation states or "
                "     adding/removing O or another balancing anion to restore neutrality."
                "  2. SUBSTITUTION: Swap one element for a chemically similar one (same group, similar ionic radius, "
                "     similar oxidation state range) if doing so is what differentiates the formula from a known "
                "     compound — e.g., swap Ca for Sr, or P for As, on a given site."
                "  3. ADDITION: Introduce a new element or ion onto an existing site (doping-style) or as a new "
                "     sublattice/co-cation/co-anion, if needed to both (a) break novelty-blocking similarity to a known "
                "     compound and (b) keep charge balance achievable."
                "  4. SUBTRACTION: Remove an element or reduce a polyatomic group to a simpler one (e.g., phosphate → "
                "     pyrophosphate fragment, or drop a water of hydration) if the resulting formula is still valid and "
                "     becomes novel as a result."
                "  5. RESTRUCTURING: Reorganize which atoms form which polyatomic sub-units entirely (e.g., re-partition "
                "     leftover atoms into a different oxyanion or a mixed-anion sublattice) if smaller changes can't "
                "     achieve novelty without breaking validity."

                "After each change, re-run the full charge-neutrality / electron-count check from Step 5 — do not carry "
                "forward a charge-balanced state from before the edit and assume it still holds."

                "═══ NOVELTY SELF-CHECK (MANDATORY) ═══"
                "Before finalizing, explicitly ask yourself: 'Is this exact formula, or an extremely close stoichiometric "
                "variant of it, a well-known mineral, salt, oxide, or compound I can recall?' If yes, you have not yet "
                "achieved novelty — go back to the operations above and make a further adjustment, then re-verify charge "
                "balance again. Do not stop at the first variant you produce; iterate until the formula is both new and "
                "valid. Trivial or superficial changes that don't actually alter the compound's identity (e.g., changing "
                "only a hydrate count, or multiplying every subscript by the same factor) do NOT count as achieving "
                "novelty and must not be presented as the final answer."

                "NOVELTY DOES NOT MEAN ARBITRARY. Do not achieve novelty by making nonsensical changes — for example, "
                "do not assign an oxidation state with no chemical precedent, do not pair elements that would violently "
                "react, and do not leave the formula charge-imbalanced. If you cannot find an adjustment that is BOTH "
                "novel AND valid using a given element set, prefer the smallest ratio/substitution change that achieves "
                "both, even if it means moving further from the original Stage 1 formula."

                "═══ OUTPUT FORMAT ═══"
                "Return ONLY: "
                "The final structural empirical formula, labeled ONLY with the words: Final Answer:, AT THE END OF YOUR RESPONSE"
                "NEVER OUTPUT ANYTHING ELSE, AND ALWAYS LABLE YOUR ANSWER"
            ),
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt} \n Empirical formula: {formula}",
        },
    ]

    return ollama_chat(messages, max_tokens=10000, temperature=0.2)


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="Logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ──────────────────────────────────────────────────────────────────────────────
# Formula parser
# ──────────────────────────────────────────────────────────────────────────────

def Sub_Beta_Function(input_text: str, key: str):
    formula = ""
    if key in input_text:
        raw_answer = input_text.split(key)[-1]
        formula = raw_answer.strip().strip('"').strip("'").strip("\u201c").strip("\u201d").strip("*")
        print("formula extracted")
        print(f"formula in parser: {formula}")
    else:
        print("Could not extract formula")
        data["payload"]["logs"]["warnings"].append("COULD NOT EXTRACT FORMULA - BETA PROCESS")
    return formula


# ──────────────────────────────────────────────────────────────────────────────
# Beta array deduplication
# ──────────────────────────────────────────────────────────────────────────────

def Beta_Array_Cleaner(Beta_Array):
    return list({entry["Formula"]: entry for entry in Beta_Array}.values())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

StabilityArray = []

if __name__ == "__main__":

    Alpha_Array = []
    Beta_Array  = []

    with open("Alpha_Arrays.json", "r", encoding="utf-8") as file:
        Alpha_Array = json.load(file)

    for i in range(len(Alpha_Array)):

        print("\n")
        print(f"{i+1} : {Alpha_Array[i]}")

        Reconstructed_Formula = (Super_Beta_Function("What is this Series of Elements original Empirical Formula:",Alpha_Array[i],).replace("#", "").replace('"', ""))

        formula = Sub_Beta_Function(Reconstructed_Formula, "Final Answer:")
        print(f"formula: {formula}")

        ScreeningResult = Beta_Function("Screen this empirical formula", formula)
        print(f"Screening results: {ScreeningResult}")
        data["payload"]["logs"]["warnings"].append(f"Screening results: {ScreeningResult} - BETA PROCESS")

        Stability_Score = Sub_Beta_Function(ScreeningResult, "Score:")
        print(f"Stability Score: {Stability_Score}")
        data["payload"]["logs"]["warnings"].append(f"Stability Score: {Stability_Score}, formula {formula} - BETA PROCESS")

        print("\n")
        print("\n")
        print("#############################################")

        try:

            if int(Stability_Score) >=  50:
                new_entry = {
                    "Index": i + 1,
                    "Formula": formula,
                    "Score": Stability_Score
                }
                Beta_Array.append(new_entry)

            time.sleep(2)
        except Exception as e:
            print(f"Error happened: {e} Skipping {formula}")
            data["payload"]["logs"]["warnings"].append(f"Error happened: {e}")
            continue

    cleaned_Beta_Array = Beta_Array_Cleaner(Beta_Array)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    data["payload"]["stage_timing"][1]["seconds"] = execution_time

    data["payload"]["candidates_in_system"] = []
    for i, entry in enumerate(Beta_Array):
        data["payload"]["candidates_in_system"].append(
            {"formula": entry["Formula"], "index": i, "id": f"cand_{i}", "status": "Passed Beta"}
        )

    with open("moonshine_data.json", "w") as f:
        json.dump(data, f, indent=2)

    with open("BetaFile.json", "w") as j:
        json.dump(cleaned_Beta_Array, j, indent=4, ensure_ascii=False)