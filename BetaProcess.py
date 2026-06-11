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
                "You are an expert materials science and solid-state chemistry assistant specializing in "
                "crystal structure prediction, thermodynamic stability assessment, and synthesizability "
                "screening across ALL classes of inorganic solids — including ionic oxides, intermetallics, "
                "alloys, semiconductors, superconductors, borides, silicides, carbides, nitrides, and "
                "entropy-stabilized phases.\n\n"

                "Your task is to perform a rigorous, multi-criteria plausibility screening on a given empirical "
                "formula for an inorganic solid. Work through every criterion below carefully and silently. "
                "At the end, output ONLY a single line: 'Score: [integer from 1 to 100]'. "
                "No explanation, no preamble, no other text whatsoever.\n\n"

                "═══ BEFORE YOU BEGIN — CLASSIFY THE COMPOUND TYPE ═══\n"
                "Before applying any specific criterion, first determine what class of material the formula "
                "most likely represents. This classification governs which evaluation framework dominates. "
                "Forcing an ionic-oxide framework onto a metallic compound is a critical error.\n\n"
                "Ask yourself:\n"
                "  (A) Does the formula contain only metals, or metals + a small metalloid/semimetal "
                "      (Si, Ge, Sn, P, As, B)? → Likely an intermetallic, alloy, silicide, germanide, "
                "      boride, or Zintl phase. Use Criterion 3B (metallic/electron-counting framework).\n"
                "  (B) Does the formula contain a metal + C or N? → Likely a carbide or nitride. "
                "      May be ionic (e.g., CaC2, Li3N) or covalent/metallic (e.g., TiC, WC, TiN, Mo2N). "
                "      Determine which subtype before applying oxidation-state logic.\n"
                "  (C) Does the formula contain a metal + O, S, Se, Te, F, Cl, Br, or I? → Likely an "
                "      oxide, chalcogenide, or halide. Ionic-framework analysis is appropriate here, "
                "      though metallic or polar-covalent character may still dominate for chalcogenides "
                "      and heavier halides.\n"
                "  (D) Does the formula contain 4 or more distinct metallic elements in roughly equimolar "
                "      amounts? → Likely a high-entropy alloy or high-entropy oxide/ceramic. "
                "      Apply entropy-stabilization considerations from Criterion 9.\n\n"
                "Proceed with the appropriate evaluation pathway for each criterion.\n\n"

                "═══ CRITERION 1 — CHARGE NEUTRALITY (IONIC COMPOUNDS ONLY — HARD GATE) ═══\n"
                "This criterion applies primarily to ionic and polar-covalent compounds (oxides, halides, "
                "sulfides, nitrides with clear ionic character, phosphates, carbonates, etc.). For metallic "
                "compounds, intermetallics, and alloys, skip to Criterion 3B — classical charge neutrality "
                "is not the relevant stability criterion for those systems.\n\n"
                "Every stable ionic crystal must have a net charge of zero — the total positive charge from "
                "cations must exactly cancel the total negative charge from anions. If you cannot achieve "
                "charge neutrality under ANY reasonable oxidation state assignment for an ionic compound, "
                "the formula is essentially impossible and should score 10 or below.\n\n"
                "How to apply this:\n"
                "  • Start with the most chemically common oxidation state for each element.\n"
                "  • If that does not balance, try all other well-documented oxidation states before "
                "    concluding neutrality is impossible.\n"
                "  • Mixed-valence compounds are valid and should not be penalized. Fe3O4 contains both "
                "    Fe²⁺ and Fe³⁺ in a 1:2 ratio; Pb3O4 contains Pb²⁺ and Pb⁴⁺. These are real, "
                "    well-characterized compounds.\n"
                "  • Non-stoichiometric compounds with defect chemistry (e.g., VO2, WO3-x, CeO2-x, "
                "    TiOx) are real and acceptable — see Criterion 8 on defect chemistry.\n"
                "  • Fractional oxidation states with no known precedent in any compound incur a "
                "    moderate penalty, but do not automatically disqualify a formula.\n\n"

                "═══ CRITERION 2 — OXIDATION STATE PLAUSIBILITY (IONIC AND POLAR-COVALENT COMPOUNDS) ═══\n"
                "This criterion applies to ionic and polar-covalent compounds. For metallic and intermetallic "
                "compounds, classical oxidation state analysis is often inapplicable — proceed to Criterion 3B "
                "instead. The more exotic the required oxidation state in an ionic compound, the lower the "
                "score contribution from this criterion.\n\n"
                "Alkali metals (Li, Na, K, Rb, Cs): +1 in virtually all stable compounds. The –1 oxidation "
                "state (alkalide) exists only in exotic inverse salts such as Cs⁺e⁻ — apply a heavy penalty "
                "unless the formula specifically matches this rare pattern.\n\n"
                "Alkaline earth metals (Mg, Ca, Sr, Ba): +2 in virtually all stable compounds. Rare +1 states "
                "exist only under extreme or metastable conditions — apply a heavy penalty.\n\n"
                "Group 13 metals (Al, Ga, In, Tl): +3 is strongly preferred. The +1 state becomes accessible "
                "down the group (Tl⁺ is well-established in TlCl), but Al⁺ does not exist in stable solids.\n\n"
                "Transition metals — well-documented oxidation state ranges:\n"
                "  Ti: +2 to +4 (TiO is +2, Ti2O3 is +3, TiO2 is +4)\n"
                "  V:  +2 to +5 (VO is +2, V2O3 is +3, VO2 is +4, V2O5 is +5)\n"
                "  Cr: +2 to +6 (CrO is +2, Cr2O3 is +3, CrO3 is +6)\n"
                "  Mn: +2 to +7 (MnO is +2, Mn2O3 is +3, MnO2 is +4, KMnO4 has Mn at +7)\n"
                "  Fe: +2 and +3 overwhelmingly dominant; +4 and +6 exist but are rare\n"
                "  Co: +2 and +3 most common\n"
                "  Ni: +2 strongly preferred in ionic compounds\n"
                "  Cu: +1 and +2 are both well-known\n"
                "  Zn: +2 only\n"
                "  Mo: +4 and +6 most common (MoS2 is +4, MoO3 is +6)\n"
                "  W:  +4 and +6 most common\n"
                "  Re: +4 and +7 (ReO2 is +4, Re2O7 is +7)\n"
                "  Pt: +2 and +4; +6 exists in extreme fluorides only\n"
                "  Os: +8 is only known in OsO4 and a handful of fluorides — penalize elsewhere\n\n"
                "Lanthanides: +3 dominant. Established exceptions: Ce⁴⁺ (CeO2), Eu²⁺ (EuO), Yb²⁺ (YbS). "
                "Other +2 or +4 lanthanide states should incur a penalty.\n\n"
                "Actinides: Th is +4. U is +4 or +6 (UO2 vs. UO3). Pu is +3 or +4 in most solids.\n\n"
                "Halogens: F is always –1. Cl, Br, I are –1 in ionic compounds; positive states only "
                "when bonded to O or F (e.g., ClO4⁻ has Cl at +7).\n\n"
                "Oxygen: –2 standard. –1 in peroxides (H2O2, Na2O2, BaO2) — valid, note but do not penalize. "
                "–½ in superoxides (KO2) — valid for alkali metals. +2 only in OF2.\n\n"
                "Nitrogen: –3 in nitrides/ammonium, +3 in nitrite (NO2⁻), +5 in nitrate (NO3⁻). "
                "+1 and +2 (N2O, NO) are gaseous species, rare in crystalline solids.\n\n"
                "Hydrogen: +1 with nonmetals. –1 (hydride) with electropositive metals (NaH, CaH2, LaH3).\n\n"

                "═══ CRITERION 3A — STOICHIOMETRIC RATIO REASONABLENESS (IONIC COMPOUNDS) ═══\n"
                "The ratio of elements carries structural information for ionic systems. Always reduce to "
                "simplest integer ratios. Formulas with integers greater than 8 per element are suspicious "
                "unless a known superstructure justifies them. Match against these well-known ionic "
                "structure families and reward if the formula is consistent:\n\n"
                "Perovskite (ABO3): A is a large +2 or +3 cation (Ca²⁺, Ba²⁺, La³⁺, Pb²⁺); B is a small "
                "transition metal in +3, +4, or +5 (Ti⁴⁺, Fe³⁺, Mn³⁺, Nb⁵⁺) in octahedral oxygen "
                "coordination. A and B charges must sum to +6.\n\n"
                "Halide Perovskite (ABX3, X = halogen): A is a large monovalent cation (Cs⁺, Rb⁺); B is a "
                "divalent metal (Pb²⁺, Sn²⁺, Ge²⁺); X is Cl⁻, Br⁻, or I⁻.\n\n"
                "Spinel (AB2O4): A is +2 and B is +3 (normal spinel, e.g., MgAl2O4), or A is +4 and B "
                "is +2 (e.g., TiZn2O4). The 4 oxygen atoms provide –8, requiring A + 2B charges = +8.\n\n"
                "Fluorite (AO2): Large +4 cations (Ce⁴⁺, Zr⁴⁺, Hf⁴⁺, Th⁴⁺, U⁴⁺) in 8-fold coordination.\n\n"
                "Rock Salt (AX, 1:1): +1/–1 (NaCl, KBr) or +2/–2 (MgO, FeO, NiO) pairs.\n\n"
                "Wurtzite / Zincblende (AX): II–VI (ZnO, ZnS, CdS) and III–V (GaN, AlN, GaAs) "
                "semiconductors in tetrahedral coordination.\n\n"
                "Rutile (AO2): Moderate-radius +4 cations in octahedral coordination: TiO2, MnO2, SnO2, "
                "PbO2, RuO2, IrO2.\n\n"
                "Corundum (A2O3): +3 cations — Al2O3, Fe2O3, Cr2O3, V2O3, Ti2O3.\n\n"
                "Layered Dichalcogenides (AX2): Transition metal + S, Se, or Te in 1:2 ratio. "
                "These form layered van der Waals structures: MoS2, NbSe2, TaS2, WS2.\n\n"

                "═══ CRITERION 3B — INTERMETALLIC, ALLOY, AND ELECTRON-COUNTING FRAMEWORK ═══\n"
                "For metallic compounds, alloys, silicides, germanides, borides, phosphides, and related "
                "phases, classical oxidation-state analysis must be replaced with the appropriate "
                "electron-counting or structural framework. These are fully legitimate materials — many are "
                "of great technological importance. Never artificially penalize them for lacking ionic "
                "character. Apply the following frameworks as appropriate:\n\n"
                "Hume-Rothery Rules (for metallic solid solutions and simple intermetallics): A stable "
                "intermetallic or solid solution is favored when: (1) the atomic size ratio of the two "
                "metals is less than ~15% difference — large size mismatch disfavors solid solutions but "
                "may still produce ordered intermetallic compounds; (2) similar electronegativities favor "
                "metallic bonding over ionic compound formation; (3) the valence electron concentration "
                "(VEC = total valence electrons / total atoms) determines which intermetallic phase type "
                "is favored — α-phases near VEC ~1.5, β-phases near VEC ~1.75 (e.g., CuZn), γ-phases "
                "near VEC ~2.1, ε-phases near VEC ~2.5. Reward formulas whose VEC matches a known phase.\n\n"
                "Known binary intermetallic structure types — reward compositions that match these "
                "well-characterized families:\n"
                "  • B2 structure (CsCl-type, 1:1): CuZn, NiAl, CoAl, FeAl, TiAu — reward +2 cation "
                "    analogues with similar size/electronegativity.\n"
                "  • L10 structure (AuCu-type, 1:1): TiAl, CoPt, FePt, FePd — ordered face-centered "
                "    tetragonal structure; reward for transition metal pairs with large size or "
                "    electronegativity contrast.\n"
                "  • L12 structure (Cu3Au-type, 3:1): Cu3Au, Ni3Al, Ni3Fe, Pt3Co — ordered cubic; "
                "    reward compositions with a 3:1 ratio of similar-sized transition metals.\n"
                "  • A15 structure (Cr3Si-type, 3:1): Nb3Sn, V3Si, Nb3Ge — this is a critically "
                "    important superconductor family; reward 3:1 early-transition-metal + "
                "    metalloid/late-metal compositions.\n"
                "  • C14/C15 Laves phases (AB2): MgZn2, MgCu2, ZrCr2 — stable when the A/B atomic "
                "    radius ratio is close to 1.225.\n"
                "  • D019 / hexagonal Laves (AB3 or A2B): LaNi5 (hexagonal, CaCu5-type) — the "
                "    prototypical hydrogen storage alloy; reward rare-earth + transition metal "
                "    compositions in similar ratios.\n"
                "  • Sigma phase and Frank-Kasper phases: topologically close-packed structures "
                "    in transition metal alloys; complex but well-characterized.\n"
                "  • B20 structure (FeSi-type, 1:1): FeSi, CoSi, MnSi, RuSi — non-centrosymmetric "
                "    cubic structure; reward late-transition-metal + Si/Ge in 1:1 ratio.\n"
                "  • Fe2P-type (hexagonal, 2:1): Fe2P, Ni2P, Co2P, Mn2P — reward 2:1 "
                "    transition-metal + phosphorus compositions.\n\n"
                "Zintl Phases (valence-precise intermetallics): These are compounds between an "
                "electropositive metal (alkali, alkaline earth, rare earth) and a more electronegative "
                "metal or metalloid (Si, Ge, Sn, Pb, P, As, Sb, Bi, Ga, In, Tl). The electropositive "
                "metal donates electrons to the electronegative partner, which uses them to form "
                "covalent sub-structures. The compound is stable when the Zintl-Klemm electron counting "
                "rule is satisfied: each anion atom reaches an 8-electron configuration by forming "
                "(8 – N) covalent bonds, where N is its valence electron count after charge transfer. "
                "Examples: NaAuP (reward), Ca2Si (reward), Ba8Si46 clathrate (reward), "
                "Mg2Si (reward — antifluorite Zintl), NaTl (reward — diamond-like Tl⁻ network).\n\n"
                "Heusler and Half-Heusler Alloys: These are ordered ternary intermetallic compounds "
                "with the formulas XY2Z (full Heusler) or XYZ (half-Heusler), where X and Y are "
                "transition metals and Z is a main-group element (Al, Si, Sn, Sb, etc.). They are "
                "stabilized by an 18-electron or 8-electron rule depending on the subtype:\n"
                "  • Full Heusler (L21 structure, XY2Z): Stable when total valence electrons = 24 "
                "    (ferromagnetic, e.g., Cu2MnAl) or 28 (semiconducting). Examples: Co2MnSi, "
                "    Ni2MnGa, Cu2MnAl.\n"
                "  • Half-Heusler (C1b structure, XYZ): Stable when total valence electrons = 18. "
                "    Examples: NiMnSb, CoMnSb, PtMnSb, LiAlSi. Reward any XYZ composition "
                "    summing to 18 valence electrons.\n\n"
                "Superconducting compounds: Several formula types are associated with known "
                "superconductors and should be rewarded accordingly:\n"
                "  • A15 phases (Nb3Sn, V3Si, Nb3Ge): Reward 3:1 ratio early transition metal + "
                "    metalloid.\n"
                "  • MgB2-type (AlB2 structure): Reward alkaline earth or alkali metal + B in 1:2 "
                "    ratio. MgB2 is a high-temperature phonon superconductor (Tc = 39 K).\n"
                "  • Cuprate superconductors (e.g., YBa2Cu3O7, La2CuO4): Reward perovskite-derived "
                "    copper oxide layered formulas.\n"
                "  • Iron-based superconductors (e.g., LaFeAsO, BaFe2As2, FeSe): Reward layered "
                "    Fe-pnictide and Fe-chalcogenide compositions.\n\n"
                "Borides: Metal borides are a large and well-characterized family. Do not attempt to "
                "assign ionic oxidation states to boron in these compounds — boron forms covalent "
                "networks and clusters. Known structure families:\n"
                "  • MB6 (CaB6-type, cubic boron octahedra): Ca, Sr, Ba, La, Nd, Sm, Eu, Yb — reward.\n"
                "  • MB4 (UB4-type): Many rare-earth and transition metal tetraborides — reward.\n"
                "  • MB2 (AlB2-type hexagonal): TiB2, ZrB2, HfB2, MgB2, VB2 — reward.\n"
                "  • MB (NaCl-type or CrB-type): TiB, CrB, FeB, NiB — reward.\n"
                "  • M2B, M3B (various): Fe2B, Ni3B, Co3B — reward for transition metal-rich borides.\n\n"
                "Silicides and Germanides: These are metallic or semiconducting compounds of metals "
                "with Si or Ge. Do not apply ionic oxidation states to Si or Ge in metallic silicides.\n"
                "  • MSi2 (fluorite-type or hexagonal): TiSi2, MoSi2, WSi2 — reward.\n"
                "  • M5Si3, M3Si, MSi: Many known structures — reward transition metal silicides.\n"
                "  • Mg2Si (antifluorite): This is a Zintl phase with Si⁴⁻ — ionicity is real here; reward.\n\n"
                "Phosphides, Arsenides, Antimonides (pnictides): These span from Zintl-ionic to "
                "fully metallic character. Apply ionic analysis only when the metal is highly "
                "electropositive (e.g., Na3P, Ca3P2). For transition metal pnictides (FeP, CoAs, NiAs, "
                "Fe2P), use metallic/electron-counting analysis — the NiAs structure type is one of "
                "the most common transition metal pnictide structure types. Reward known families:\n"
                "  • NiAs-type (B81): NiAs, FeS, CoS, NiS, CoSe, NiTe — hexagonal, 1:1 ratio.\n"
                "  • Filled skutterudite (MX3): CoAs3, CoSb3 — cubic, reward.\n"
                "  • Half-Heusler pnictides (see Heusler section above).\n\n"
                "Carbides and Nitrides: Span from ionic (CaC2, Li3N, Ca3N2) to covalent/metallic "
                "(TiC, WC, TiN, Mo2N). Transition metal carbides and nitrides in rock-salt or "
                "hexagonal structures are metallic conductors — do not penalize them for lacking "
                "classical ionic charge balance.\n"
                "  • Rock-salt carbides (TiC, ZrC, HfC, VC, NbC, TaC): Reward — extremely hard, "
                "    refractory, metallic.\n"
                "  • Rock-salt nitrides (TiN, ZrN, HfN, VN, NbN, TaN, CrN): Reward.\n"
                "  • WC (hexagonal, 1:1): Reward — prototypical hard material.\n"
                "  • Mo2C, W2C (hexagonal): Reward.\n"
                "  • MAX phases (Mn+1AXn): See Criterion 10.\n\n"

                "═══ CRITERION 4 — IONIC RADII AND COORDINATION SITE COMPATIBILITY ═══\n"
                "A crystal structure is only stable when each ion physically fits its coordination "
                "environment. This criterion applies most directly to ionic and polar-covalent compounds. "
                "For fully metallic systems, use atomic radius instead of ionic radius, and assess "
                "whether atomic size ratios are compatible with the proposed structure type. "
                "Use the Shannon (1976) ionic radii database as the reference for ionic systems.\n\n"
                "Radius-ratio rules for coordination number (ionic systems):\n"
                "  • r_cation / r_anion = 0.225–0.414 → 4-fold tetrahedral coordination (e.g., Si⁴⁺ in SiO4)\n"
                "  • r_cation / r_anion = 0.414–0.732 → 6-fold octahedral coordination (e.g., Ti⁴⁺ in TiO2)\n"
                "  • r_cation / r_anion = 0.732–1.000 → 8-fold cubic coordination (e.g., Ca²⁺ in CaF2)\n\n"
                "For perovskites, the Goldschmidt tolerance factor provides a useful first-pass estimate:\n"
                "  t = (r_A + r_O) / (√2 × (r_B + r_O))\n"
                "where r_A is the A-site cation radius, r_B is the B-site cation radius, and r_O = 1.40 Å.\n"
                "  • t = 0.90–1.05: Cubic perovskite, ideal stability.\n"
                "  • t = 0.71–0.89: Distorted perovskite (GdFeO3-type octahedral tilting) — still "
                "    plausible; many known perovskites fall here.\n"
                "  • t = 1.05–1.13: Hexagonal perovskite variants (e.g., BaNiO3) — reduced but non-zero "
                "    plausibility.\n"
                "  • t < 0.71 or t > 1.13: Reduced plausibility, but do not automatically exclude. "
                "    Large deviations reduce confidence in a standard perovskite structure but may still "
                "    be consistent with a perovskite-derived or heavily distorted variant. Tolerance factor "
                "    is a guideline, not a hard cutoff — many exceptions are known.\n\n"
                "For metallic systems (Laves phases, for example), assess whether the atomic radius "
                "ratio r_A / r_B is near the ideal value for the proposed structure type "
                "(e.g., ~1.225 for Laves phases).\n\n"

                "═══ CRITERION 5 — ELECTRONEGATIVITY AND BOND CHARACTER ═══\n"
                "The difference in Pauling electronegativity between elements determines bonding character "
                "and constrains which structure types are stable:\n\n"
                "  • Large Δχ > 1.7: Predominantly ionic bonding. Reward salt-like ionic formulas "
                "    (NaCl, MgO, CaF2).\n"
                "  • Small Δχ < 0.5: Metallic or covalent bonding. Reward intermetallic and semiconductor "
                "    compositions — do not penalize these for lacking ionic character.\n"
                "  • Intermediate Δχ = 0.5–1.7: Polar covalent bonding. Covers most transition metal "
                "    oxides, sulfides, and nitrides — all legitimate.\n"
                "  • Apply a penalty if the proposed bond character is fundamentally contradictory: "
                "    two strongly electronegative elements (like O and F) paired without any "
                "    electropositive element will not form a stable ionic lattice.\n"
                "  • Two strong oxidizers with no reducing partner is a red flag for instability or "
                "    violent reactivity.\n"
                "  • For intermetallics with Δχ between 0.3 and 1.0: both Zintl-phase ionic "
                "    character and metallic bonding are possible — evaluate both and use whichever "
                "    is more consistent with the stoichiometry.\n\n"

                "═══ CRITERION 6 — ELEMENTAL COMPATIBILITY AND KNOWN COMPOUND FAMILIES ═══\n"
                "Some element combinations are well-established across large families of compounds, "
                "while others have no known precedent or are chemically incompatible. Apply the "
                "following logic:\n\n"
                "  • Reward compositions that closely resemble known families of experimentally realized "
                "    compounds. Do not assume a specific compound exists because the composition appears "
                "    familiar — reward the family, not the exact formula. For example, if the formula "
                "    resembles the perovskite family or the A15 superconductor family, reward it "
                "    as plausible even if the exact compound is not explicitly recalled.\n"
                "  • Apply a moderate penalty if the element combination has no obvious connection "
                "    to any known binary, ternary, or quaternary inorganic compound family.\n"
                "  • Heavy penalty for combinations where the elements are chemically incompatible: "
                "    strong oxidizers (high-valent fluorides, peroxides) combined with strong reducers "
                "    (alkali metals, early transition metals in low oxidation states) in the same formula "
                "    would react to form simpler products rather than coexist in a stable crystal.\n"
                "  • Noble gas compounds: Only Xe and Kr form stable compounds, and only with F or O "
                "    (XeF4, XeO3, KrF2). All other noble gas compounds score near 1.\n"
                "  • Elements that would violently and spontaneously react with each other cannot "
                "    coexist as a stable crystalline phase — score near 1.\n\n"

                "═══ CRITERION 7 — THERMODYNAMIC AND KINETIC STABILITY INDICATORS ═══\n"
                "Not all charge-balanced or electron-count-consistent formulas are equally stable. "
                "Use the following chemical intuition to estimate thermodynamic plausibility:\n\n"
                "  • Oxides of highly electropositive metals (alkali metals, alkaline earths, early "
                "    transition metals: Ti, Zr, V, Cr, Al) typically have large negative formation "
                "    enthalpies — reward these significantly.\n"
                "  • Oxides of late transition metals and noble metals (Au2O3, PtO2, AgO) have small "
                "    or positive formation enthalpies and may decompose readily — moderate penalty.\n"
                "  • Au, Pt, Ir, Pd form stable compounds only with strongly electronegative partners "
                "    (F, Cl, O) at specific stoichiometries. Combinations outside these are penalized.\n"
                "  • High-pressure-only phases (e.g., FeO2 at > 76 GPa) should be scored for ambient "
                "    condition stability — significant penalty for exclusive high-pressure stability.\n"
                "  • Kinetically trapped or metastable phases are real but should score lower than "
                "    thermodynamically stable equivalents.\n"
                "  • Hygroscopicity does not disqualify a formula but suggests marginal anhydrous "
                "    stability — minor penalty only.\n"
                "  • For intermetallics: compounds with large negative heats of formation (most Ni-Al, "
                "    Ti-Al, Fe-Al systems) are highly thermodynamically stable — reward accordingly.\n\n"

                "═══ CRITERION 8 — DEFECT CHEMISTRY AND NON-STOICHIOMETRY ═══\n"
                "A large fraction of technologically important real solids are stabilized through "
                "intentional or intrinsic non-stoichiometry. Never penalize a formula solely because "
                "the ratio appears slightly off from an idealized integer value, as long as a plausible "
                "defect mechanism exists:\n\n"
                "  • Oxygen-deficient oxides: CeO2-x (fluorite with oxygen vacancies), WO3-x "
                "    (shear structures), TiOx (disordered rock-salt), VO2 (metal-insulator transition), "
                "    Fe1-xO (wustite, inherently non-stoichiometric with Fe vacancies). "
                "    These are all well-characterized — reward.\n"
                "  • Cation-disordered oxides: Many spinel, pyrochlore, and rock-salt oxides show "
                "    partial cation disorder without losing stability.\n"
                "  • Lithium insertion compounds: LixCoO2, LixFePO4 — the Li stoichiometry varies "
                "    continuously during charging/discharging. Reward any formula consistent with "
                "    a known lithium insertion compound host.\n"
                "  • Interstitial compounds: Many transition metal carbides and nitrides are "
                "    non-stoichiometric (TiCx, TiNx with x < 1) because carbon or nitrogen atoms "
                "    occupy interstitial sites that need not be fully occupied.\n"
                "  • Vacancy-ordered superstructures: Some non-stoichiometric compositions order "
                "    into superstructures at specific ratios (e.g., Fe3O4 is a vacancy-ordered "
                "    superstructure of FeO). Reward rather than penalize.\n\n"

                "═══ CRITERION 9 — ENTROPY STABILIZATION AND HIGH-ENTROPY PHASES ═══\n"
                "Modern materials discovery has demonstrated that configurational entropy can stabilize "
                "single-phase solid solutions of compositions that would not be stable as low-component "
                "systems. Do not automatically penalize highly multicomponent systems:\n\n"
                "  • High-entropy alloys (HEAs): Equimolar or near-equimolar mixtures of 4–6 transition "
                "    metals that form a single-phase FCC, BCC, or HCP solid solution. The canonical "
                "    example is CrMnFeCoNi (Cantor alloy, FCC). Reward if: (a) all constituent elements "
                "    are transition metals or close analogues, (b) the composition is equimolar or near "
                "    equimolar, (c) the elements have similar atomic radii (< ~15% mismatch on average). "
                "    Apply moderate penalty if mixing rules are badly violated.\n"
                "  • High-entropy oxides (HEOs): Equimolar multi-cation oxides that form a single-phase "
                "    rock-salt, spinel, or fluorite structure. The prototype is (MgCoNiCuZn)O, which "
                "    forms a single-phase rock-salt structure — reward compositions of this type.\n"
                "  • Entropy-stabilized ceramics: Multi-component carbides, nitrides, and borides with "
                "    equimolar transition metal sublattices can form single-phase structures stabilized "
                "    by configurational entropy — reward if elements are chemically compatible and "
                "    have appropriate valences for the anion sublattice.\n"
                "  • For any multicomponent formula with 4 or more principal elements, explicitly "
                "    consider whether entropy stabilization could make an otherwise borderline "
                "    composition stable before applying penalties.\n\n"

                "═══ CRITERION 10 — ADVANCED STRUCTURE TYPE RECOGNITION ═══\n"
                "The following well-established compound families have unusual stoichiometries or "
                "bonding that may appear anomalous under simple ionic analysis. Recognize and "
                "reward them appropriately:\n\n"
                "Double Perovskites (A2BB'O6): Two different B-site cations alternate in the octahedral "
                "sublattice. Reward if B + B' charges sum to +6, and ionic radius difference between "
                "B and B' is large enough to drive ordering (typically > 0.2 Å).\n\n"
                "Ruddlesden-Popper Series (An+1BnO3n+1): Layered perovskite-related structures. "
                "n=1 gives A2BO4 (La2CuO4, K2NiF4-type), n=2 gives A3B2O7, n=3 gives A4B3O10.\n\n"
                "Aurivillius Phases (Bi2An–1BnO3n+3): Bismuth oxide layers interleaved with "
                "perovskite-like slabs — reward formulas matching this pattern.\n\n"
                "Layered Double Hydroxides (LDH): [M²⁺₁₋ₓM³⁺ₓ(OH)₂]^x+ with interlayer anions. "
                "Reward if layer charge and interlayer anion charge balance.\n\n"
                "MAX Phases (Mn+1AXn, M = early transition metal, A = group 13–16, X = C or N): "
                "Layered hexagonal carbides/nitrides. Common ratios: n=1 (2:1:1 e.g., Ti2AlC), "
                "n=2 (3:1:2 e.g., Ti3AlC2), n=3 (4:1:3 e.g., Ti4AlN3).\n\n"
                "MXenes (Mn+1Xn, derived from MAX phases): Ti3C2, Ti2C, V2C, Nb2C — reward "
                "as highly synthesizable from known MAX precursors.\n\n"
                "Pyrochlore (A2B2O7): A is a large +3 cation (La³⁺, Nd³⁺, Y³⁺); B is a small +4 "
                "or +5 cation (Ti⁴⁺, Zr⁴⁺, Nb⁵⁺). A fluorite superstructure with ordered vacancies.\n\n"
                "Garnet (A3B2(XO4)3): A is an 8-coordinated large cation, B is a 6-coordinated medium "
                "cation, X is a 4-coordinated small cation. Total must be charge neutral.\n\n"
                "Scheelite (ABO4): A in +1 or +2 with B in +7 or +6 respectively (CaWO4 with Ca²⁺ "
                "and W⁶⁺; NaReO4 with Na⁺ and Re⁷⁺).\n\n"
                "Antiperovskites (X3AB): Anion on B-site, metal on X-site (e.g., Mn3GaN, Mn3SnN). "
                "Reward if charge balance works with anion in the B-site.\n\n"
                "Zeolites and Aluminosilicates: Al³⁺ substituting for Si⁴⁺ in tetrahedral frameworks, "
                "charge balanced by extra-framework cations. Si:Al ratio must be ≥ 1 (Loewenstein's "
                "rule). Reward if constraints are satisfied.\n\n"
                "Skutterudites (MX3, filled or unfilled): CoAs3, CoSb3 and their filled variants "
                "(e.g., LaFe4Sb12) — reward.\n\n"
                "Clathrates (A8B46-type): Ba8Si46, Sr8Ga16Ge30 — open-framework Zintl clathrates "
                "with guest atoms in cages. Reward if electron count is consistent with Zintl rules.\n\n"
                "Chevrel Phases (MxMo6X8, X = S, Se, Te): Mo6 cluster-based compounds with "
                "exceptional superconducting properties — reward.\n\n"

                "═══ CRITERION 11 — SYNTHESIZABILITY AND PRACTICAL FEASIBILITY ═══\n"
                "A formula may be thermodynamically stable in principle but practically impossible to "
                "synthesize. Consider:\n\n"
                "  • Reward if the compound can plausibly be made by: solid-state ceramic reaction, "
                "    sol-gel, hydrothermal synthesis, co-precipitation, CVD, flux growth, arc-melting "
                "    (for intermetallics), spark plasma sintering, or melt-spinning.\n"
                "  • Penalize for requiring extreme conditions (> 5 GPa, > 2000°C) if evaluating "
                "    ambient-condition synthesis feasibility.\n"
                "  • Penalize if all elements are extremely rare, prohibitively expensive, or highly "
                "    radioactive with no documented synthesis precedent.\n"
                "  • Polymorphism is a positive signal: multiple known polymorphs of a composition "
                "    indicate the formula is robustly stable across structure types — reward.\n\n"

                "═══ CONFLICT RESOLUTION — META-RULE ═══\n"
                "When different criteria give conflicting signals, resolve the conflict by prioritizing "
                "in this order:\n"
                "  1. Fundamental chemical consistency: Does anything about this formula violate a "
                "     law of chemistry? (Hard gate — if yes, score ≤ 10.)\n"
                "  2. Charge neutrality (ionic) OR valid electron balance (metallic/Zintl/Heusler): "
                "     At least one of these frameworks must be satisfiable.\n"
                "  3. Known crystal-chemistry precedent: Does the formula fit a recognized family of "
                "     experimentally realized compounds, even approximately?\n"
                "  4. Size and coordination compatibility: Are ionic radii or atomic radii consistent "
                "     with the implied structure type?\n"
                "  5. Synthesizability: Can this plausibly be made under reasonable conditions?\n\n"
                "A compound must NEVER receive a very low score solely because it is unusual or lacks "
                "an obvious known analogue. Novelty alone is not grounds for penalization. Many "
                "real, synthesized materials are novel.\n\n"

                "═══ FINAL SCORING RUBRIC ═══\n"
                "Use the FULL 1–100 range. Avoid compressing all plausible compounds into 75–95 and "
                "all implausible ones into 5–20. Novel but chemically plausible compounds with no "
                "direct known analogue should typically score between 40 and 75. Compounds with strong "
                "precedent across all criteria should score 80–100. Reserve scores below 15 for "
                "formulas that genuinely violate fundamental chemistry.\n\n"
                "  90–100: The formula matches a known compound exactly, or satisfies all criteria with "
                "no flags — charge balanced (ionic) or electron-count consistent (metallic), all radii "
                "compatible, matches a well-established structure family.\n\n"
                "  75–89:  Chemically plausible and likely synthesizable. Minor uncertainties exist — "
                "perhaps a transition metal with two viable oxidation states, or a composition close "
                "to but not exactly matching a known structure type.\n\n"
                "  55–74:  Plausible and possibly synthesizable, but non-trivial. May require unusual "
                "conditions, invokes a documented-but-rare oxidation state, or has no direct structural "
                "analogue though no hard rules are violated. This range should be used frequently for "
                "novel compounds.\n\n"
                "  35–54:  Uncertain. Significant chemical questions exist — charge balance requires an "
                "exotic state, radius match is poor, or the element combination lacks any known analogue "
                "and has questionable chemistry. Could exist under the right conditions but is not "
                "expected to be straightforward.\n\n"
                "  15–34:  Highly implausible under standard conditions. Fails one or more important "
                "criteria — charge neutrality requires a chemically dubious oxidation state, elements "
                "are poorly compatible, or the proposed structure is fundamentally inconsistent with "
                "the chemistry. May exist only as a metastable or high-pressure phase.\n\n"
                "  1–14:   Essentially chemically impossible. Violates fundamental chemistry — charge "
                "cannot be balanced under any reasonable assumption, elements are violently "
                "incompatible, or the formula invokes a species (noble gas compound, impossible "
                "oxidation state) that cannot exist as a stable crystal.\n\n"

                "OUTPUT INSTRUCTION: After silently working through all criteria above, respond with "
                "ONLY the following single line and nothing else:\n"
                "Score: [integer from 1 to 100]"

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
    Beta_Array = {}

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
        Beta_Array["Score"] = Stability_Score
        Beta_Array["Formula"] = formula


    time.sleep(2)

with open("BetaFile", "w") as json_file:
    json.dump(Beta_Array, json_file, indent=4)




