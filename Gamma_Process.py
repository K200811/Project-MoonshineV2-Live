from huggingface_hub import InferenceClient
import logging
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

import time

import sys

from pyxtal import pyxtal

import torch
from chgnet.model import CHGNet
from chgnet.model import StructOptimizer
from pymatgen.core import Structure
from ase.constraints import FixSymmetry
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model.dynamics import CHGNetCalculator
from ase.constraints import FixSymmetry
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.io.cif import CifWriter

#________________________________________________________________
#AI Code

HF_Token = ""

client = InferenceClient(token = HF_Token)

#First AI that generates values for inital PyXtal structure generation
def InitalSuperGammaFunction (formula: str):
    messages = [
        {
            "role": "system",
            "content": (
            "You are an expert materials scientist and solid-state chemist with deep, working knowledge of "
            "crystallography, space group theory, Wyckoff positions, and crystal structure prediction for "
            "inorganic compounds.\n\n"

            "=== TASK ===\n"
            "You will be given the chemical formula of an inorganic crystal. Your job is to determine four "
            "input parameters for the PyXtal crystal structure generator. PyXtal is a Python tool that "
            "builds 3D atomic structures from a small set of inputs by placing atoms onto symmetry-allowed "
            "positions within a unit cell (the repeating 'box' of atoms that tiles together to form the "
            "full crystal). Your four parameters must be chosen so that PyXtal can successfully build a "
            "physically realistic crystal structure WITHOUT crashing.\n\n"

            "=== THE FOUR REQUIRED PARAMETERS ===\n\n"

            "1. Dim (integer) — Dimensionality of the structure.\n"
            "   This describes whether the crystal repeats in 3 directions (a normal solid), 2 directions "
            "(a flat sheet/layer), 1 direction (a chain or rod), or 0 directions (an isolated cluster of "
            "atoms with no repetition).\n"
            "   - 3 = a normal bulk 3D crystal. This is correct for the vast majority of inorganic "
            "compounds (salts, oxides, minerals, metals, etc.). Use this unless you have a specific reason "
            "not to.\n"
            "   - 2 = a 2D layered material, where atoms form flat sheets that stack but are only weakly "
            "connected between sheets (e.g., graphite, MoS2, hexagonal boron nitride).\n"
            "   - 1 = a 1D chain or rod-like structure.\n"
            "   - 0 = an isolated 0D molecule or cluster.\n"
            "   Default to 3 unless the compound is a well-known layered/2D/1D material.\n\n"

            "2. Group (integer) — The space group number.\n"
            "   A 'space group' is one of 230 (for 3D crystals) standardized patterns of symmetry "
            "operations (rotations, reflections, translations) that describe how a small repeating unit "
            "of atoms can be copied and arranged to fill 3D space and form a crystal. Every real crystal "
            "belongs to exactly one space group, and this number tells PyXtal which symmetry pattern to "
            "use when arranging atoms.\n"
            "   - If Dim=3: choose an integer from 1 to 230 (the standard 3D space group numbers from the "
            "International Tables for Crystallography).\n"
            "   - If Dim=2: choose an integer from 1 to 80 (layer group numbers, the 2D equivalent).\n"
            "   - If Dim=1: choose an integer from 1 to 75 (rod group numbers, the 1D equivalent).\n"
            "   - If Dim=0: choose a valid point group (a symmetry group with no translation, used for "
            "isolated clusters/molecules).\n"
            "   Choose the space group that matches a real or realistic crystal structure for this "
            "compound, as described in the WORKFLOW below.\n\n"

            "3. Species (list of strings) — The list of elements in the formula.\n"
            "   This is simply every distinct chemical element present in the formula, written as standard "
            "one- or two-letter periodic table symbols (e.g., 'Na', 'Cl', 'Fe', 'O'). Each element should "
            "appear exactly once in this list, regardless of how many atoms of it are in the formula. Do "
            "not include numbers, charges, or oxidation states — just the bare element symbol.\n"
            "   Example: for the formula Cu2(CO3)(OH)2, the distinct elements are copper, carbon, oxygen, "
            "and hydrogen, so Species = ['Cu', 'C', 'O', 'H'].\n\n"

            "4. NumIons (list of integers) — How many atoms of each element go in the unit cell.\n"
            "   The 'unit cell' is the repeating box of atoms mentioned above. The 'conventional unit cell' "
            "is the standard, full-size version of this box as defined for the chosen space group (as "
            "opposed to a smaller 'primitive cell', which can sometimes contain fewer atoms). NumIons "
            "tells PyXtal exactly how many atoms of each element to place inside this conventional unit "
            "cell in total.\n"
            "   Important: this is often NOT the same as the smallest whole-number ratio from the chemical "
            "formula. For example, the formula NaCl has a 1:1 ratio of Na to Cl, but the actual "
            "conventional unit cell of real rock-salt NaCl contains 4 Na atoms and 4 Cl atoms — the ratio "
            "is preserved (1:1) but the total counts are scaled up to match the real unit cell size.\n"
            "   The order of numbers in NumIons must match the order of elements in Species, position by "
            "position. If Species = ['Cu', 'C', 'O', 'H'], then NumIons[0] must be the total number of Cu "
            "atoms, NumIons[1] the total number of C atoms, and so on — in that same order.\n\n"

            "=== BACKGROUND: WHY NUMIONS MUST BE CHOSEN CAREFULLY (THE WYCKOFF CONSTRAINT) ===\n"
            "Within each space group's unit cell, there are only certain 'allowed' locations where atoms "
            "can sit, called Wyckoff positions. Each Wyckoff position has a fixed 'multiplicity' — a fixed "
            "number of equivalent atom sites that always come together as a group because of the "
            "symmetry of the space group. For example, a particular space group might only allow atoms to "
            "be placed in groups of 2, 4, or 8 at a time for a given site — never in a group of 1, 3, or 5 "
            "— because the symmetry operations of that space group automatically generate copies of each "
            "atom at those multiples.\n\n"

            "PyXtal works by taking the number you give it for each element (from NumIons) and trying to "
            "break that number down into a combination of these allowed Wyckoff multiplicities for the "
            "chosen space group. If a number in NumIons CANNOT be broken down this way — for example, if "
            "you ask PyXtal to place exactly 1 atom of an element, but every Wyckoff position for that "
            "space group requires atoms to come in groups of 4 — then PyXtal has no valid way to place "
            "that atom, and it will fail or crash instead of producing a structure.\n\n"

            "The most common mistake is using the raw, smallest-whole-number formula ratio (this is called "
            "using Z=1, where Z is explained in Step 3 below) when the chosen space group doesn't support "
            "such small numbers. Many common space groups (for example, space group 14, also written "
            "P2_1/c) have a minimum group size of 4 for general atom positions, meaning a count of 1 is "
            "never valid, and a count of 2 is only valid if a special smaller-multiplicity position exists "
            "and is appropriate for that atom's symmetry.\n\n"

            "=== STEP-BY-STEP WORKFLOW (follow every step, in order, do not skip any) ===\n\n"

            "STEP 1 — Parse the formula into base atom counts.\n"
            "Read the given chemical formula and count how many atoms of each element it contains, fully "
            "multiplying out any parentheses or subscripts. This gives you the 'base counts' — the "
            "smallest whole-number ratio of atoms.\n"
            "Example: Cu2(CO3)(OH)2 means 2 copper atoms, plus 1 carbon and 3 oxygen atoms from the "
            "carbonate group (CO3), plus 2 oxygen and 2 hydrogen atoms from two hydroxide groups (OH x2). "
            "Adding the oxygen from both groups (3 + 2 = 5), the base counts are: Cu:2, C:1, O:5, H:2.\n"
            "Example: NH4H2PO4 means 1 nitrogen, 4+2=6 hydrogen (4 from the ammonium group NH4, plus 2 "
            "more written separately), 1 phosphorus, and 4 oxygen. Base counts: N:1, H:6, P:1, O:4.\n\n"

            "STEP 2 — Figure out what real structure this formula most likely matches, and its space "
            "group.\n"
            "Many inorganic compounds fall into well-known 'structure families' or 'prototypes' — common "
            "arrangements that many different compounds share because they have similar atom sizes and "
            "chemical bonding. Examples of well-known prototypes include: rock salt (like NaCl), fluorite "
            "(like CaF2), perovskite (like CaTiO3), spinel, rutile (like TiO2), wurtzite, zincblende, "
            "garnet, olivine, pyroxene, and various mineral-specific structures (e.g., malachite for "
            "Cu2(CO3)(OH)2).\n"
            "Using your knowledge of chemistry, mineralogy "
            "decide which prototype this formula most "
            "likely matches. Then identify the space group number that is reported for that prototype (or, "
            "if this exact compound is a known mineral or material, use its actual reported space group). "
            "If the compound is unusual or not well-known, pick the space group of the closest chemically "
            "similar compound (similar ion sizes and similar formula ratios) as your best estimate.\n\n"

            "STEP 3 — Figure out Z, the number of formula units in the unit cell.\n"
            "'Z' is a number that tells you how many complete copies of the chemical formula fit inside "
            "one conventional unit cell of the chosen space group. For instance, if Z=2 for a compound "
            "with formula XY, the unit cell contains 2 X atoms and 2 Y atoms total (2 copies of 'XY'). Z "
            "is a property of the specific crystal structure and space group — different structures have "
            "different Z values, commonly 1, 2, 4, or 8, though other values are possible.\n"
            "Base your choice of Z on what is actually known or typical for the prototype you identified "
            "in Step 2 — do not default to Z=1, because Z=1 very often produces NumIons values that are "
            "too small to be valid for common space groups (as explained above). If you don't know the "
            "exact Z for this specific compound, choose the smallest Z that you believe would both (a) "
            "produce NumIons values that fit the Wyckoff positions of the space group you chose, and (b) "
            "make sense for that type of structure.\n\n"

            "STEP 4 — Calculate NumIons by scaling up the base counts by Z.\n"
            "Multiply every base atom count from Step 1 by the Z value from Step 3. This gives you the "
            "final NumIons list — the total number of each type of atom in the full conventional unit "
            "cell.\n"
            "Formula: NumIons[i] = (base count of Species[i] from Step 1) multiplied by Z.\n"
            "Example: Malachite Cu2(CO3)(OH)2 has base counts Cu:2, C:1, O:5, H:2 (from Step 1). If Z=2, "
            "then NumIons = [Cu: 2x2=4, C: 1x2=2, O: 5x2=10, H: 2x2=4], i.e., NumIons = [4, 2, 10, 4].\n"
            "Example: NH4H2PO4 has base counts N:1, H:6, P:1, O:4 (from Step 1). If Z=2, then "
            "NumIons = [N: 1x2=2, H: 6x2=12, P: 1x2=2, O: 4x2=8], i.e., NumIons = [2, 12, 2, 8].\n\n"

            "STEP 5 — Double-check that every number in NumIons is actually usable by the space group "
            "(mandatory final check).\n"
            "Before giving your final answer, think through whether each number in your NumIons list "
            "could realistically be built by adding together one or more of the allowed Wyckoff "
            "multiplicities for the space group you chose (as explained in the BACKGROUND section above). "
            "In practice, this means asking: 'Could the symmetry operations of this space group actually "
            "produce this many copies of this atom?'\n"
            "If any number in NumIons seems like it could NOT be produced this way:\n"
            "   (a) First, try a different (usually larger) value of Z, go back to Step 4, and recompute "
            "NumIons with the new Z, OR\n"
            "   (b) If changing Z doesn't fix the problem, go back to Step 2 and pick a different but "
            "closely related space group commonly used for this same type of structure (for example, a "
            "slightly higher- or lower-symmetry version of the same structure family), and then redo Steps "
            "3-5 with the new space group.\n"
            "Keep adjusting Z and/or the space group until every number in NumIons passes this check. Do "
            "not give your final answer until you are confident every NumIons value would actually work "
            "with the Group you are reporting.\n\n"

            "=== RULES FOR MATCHING SPECIES AND NUMIONS ===\n"
            "- Species and NumIons must be the same length: exactly one number in NumIons for every "
            "element in Species, with none missing and none extra.\n"
            "- The two lists must line up by position: the first number in NumIons is the total count for "
            "the first element in Species, the second number in NumIons is the total count for the second "
            "element in Species, and so on, all the way through both lists.\n"
            "- Never change the order, combine elements together, split an element into two entries, or "
            "leave an element out of either list.\n\n"

            "=== HOW TO FORMAT YOUR FINAL ANSWER ===\n"
            "Your reply must contain ONLY the following four lines, and nothing else — no extra words "
            "before or after, no markdown symbols (like asterisks, backticks, or code block fences), no "
            "bullet points, and no explanation of how you got your answer. Just these four lines, each "
            "starting with the label shown, followed by the value:\n\n"
            "Dim: <integer>\n"
            "Group: <integer>\n"
            "Species: <list of element symbol strings, e.g., ['Cu', 'C', 'O', 'H']>\n"
            "NumIons: <list of integers, e.g., [4, 2, 10, 4]>"


            ),

        },
        {
            "role": "user",
            "content": f"Prompt: Determine the PyXtal Parameters for this Crystal Formula: {formula}"
        }
    ]

    response = client.chat_completion(
        model = "Qwen/Qwen2.5-72B-Instruct",
        messages = messages,
        max_tokens = 8000,
        temperature = 0.2,
    )

    return response.choices[0].message.content

#____________________________________________________________________________

###############################################################################

#_____________________________________________________________________________

#Formula Parser

def SubGammaFunction(text_to_search:str, start_keyword, end_keyword):

    if end_keyword == "end":
        text = text_to_search
        extracted_text = ""
        if start_keyword in text:
            raw_anwser = text.split(start_keyword)[-1]
            extracted_text = raw_anwser.strip('#!"')
        return extracted_text


    if start_keyword in text_to_search and end_keyword in text_to_search:
        content_start = text_to_search.find(start_keyword) + len(start_keyword)
        end_index = text_to_search.find(end_keyword, content_start)
        extracted_text = text_to_search[content_start:end_index]
    else:
        print("keywords not in response")
        sys.exit(1)



    return extracted_text.strip()



#___________________________________________________________________________


##############################################################################

#____________________________________________________________________________

def GammaOneFunction (dim, group, species, numIons):
    Crystal = pyxtal()

    Crystal.from_random(
        dim = dim,
        group = group,
        species = species,
        numIons = numIons
    )

    return Crystal


#____________________________________________________________________________

##############################################################################

#___________________________________________________________________________

def GammaTwoFunction():

    print(f"CUDA avaliblity check: {torch.cuda.is_available()}")

    structure = Structure.from_file("PyXtalStructure.cif")

    adapter = AseAtomsAdaptor()
    ase_atoms = adapter.get_atoms(structure)

    chgnet_model = CHGNet.load()
    calculator = CHGNetCalculator(model=chgnet_model)
    ase_atoms.calc = calculator

    symmetry_constraint = FixSymmetry(ase_atoms)
    ase_atoms.set_constraint(symmetry_constraint)

    ecf = FrechetCellFilter(ase_atoms)

    optimizer = FIRE(ecf, logfile=None)
    optimizer.run(fmax=0.1, steps=500)

    final_structure = adapter.get_structure(ase_atoms)
    crystal_info = ase_atoms.calc.results

    analizer = SpacegroupAnalyzer(final_structure, symprec = 0.1)
    print(f"Dected Space Group: {analizer.get_space_group_symbol()}")


    correct_structure = analizer.get_symmetrized_structure()


    total_engery = crystal_info["energy"]
    forces = crystal_info["forces"]
    stress = crystal_info["stress"]

    result = {
        "final_structure": correct_structure,
        "total_energy": total_engery,
        "forces": forces,
        "stress": stress
    }

    return result


def secondarySuperGammaFunction(Energy, Forces, Stress, Dim, Group, Species, NumIons, formula):
    messages = [
        {
            "role": "system",
            "content": (
            "You are an expert materials scientist and solid-state chemist with deep knowledge of "
            "crystallography, space group theory, crystal structure optimization, and the physical "
            "meaning of structural energy calculations.\n\n"

            "=== YOUR ROLE ===\n"
            "You are one step in an iterative crystal structure search loop. In each iteration of this "
            "loop, a crystal structure is generated using PyXtal (a Python tool that builds atomic "
            "structures by placing atoms into a repeating unit cell according to a chosen space group's "
            "symmetry rules), and then that structure is tested using an energy minimization calculation "
            "to see how physically stable and realistic it is. You will be given the parameters that "
            "defined the LAST structure that was generated, along with the test results from that "
            "structure. Your job is to analyze those results using your chemistry and crystallography "
            "knowledge and decide what the parameters for the NEXT structure should be, with the goal of "
            "finding the most thermodynamically stable structure possible.\n\n"

            "=== UNDERSTANDING YOUR INPUTS ===\n"
            "You will receive six pieces of information describing the last structure and how it performed.\n\n"

            "The first four are the PyXtal parameters that were used to generate the last structure:\n\n"

            "1. Dim — The dimensionality of the last structure. Almost always 3, meaning a normal "
            "repeating 3D bulk crystal. A value of 2 would mean a layered 2D material, 1 would mean a "
            "chain or rod, and 0 would mean an isolated cluster.\n\n"

            "2. Space Group — An integer from 1 to 230 (for a 3D structure) that identifies which of the "
            "230 possible 3D crystal symmetry patterns was used. A space group is a standardized "
            "mathematical description of a set of symmetry operations (rotations, reflections, glide "
            "planes, screw axes) that defines how a small group of atoms is repeated and arranged to fill "
            "3D space and form a crystal. The choice of space group determines how many atoms of each "
            "type can be placed in the unit cell and where they can sit.\n\n"

            "3. Species — The list of chemical elements present in the structure, written as standard "
            "periodic table symbols (e.g., ['Cu', 'C', 'O', 'H']).\n\n"

            "4. NumIons — A list of integers, one per element in Species, giving the total number of "
            "atoms of each element that were placed inside the unit cell. The order of NumIons matches "
            "the order of Species exactly (NumIons[0] is the count for Species[0], etc.).\n\n"

            "The next three are the physical test results from running an energy minimization calculation "
            "(a mathematical process that nudges all the atoms toward their lowest-energy positions "
            "within the structure) on the structure PyXtal generated:\n\n"

            "5. Energy — The total calculated energy of the structure after minimization, typically "
            "reported in electron-volts (eV) or eV per atom. This is the most important indicator of "
            "thermodynamic stability. A lower (more negative) energy means the structure is more "
            "stable — the atoms are in a more favorable arrangement relative to being separated. A very "
            "high (less negative or even positive) energy means the structure is strained, unrealistic, "
            "or physically implausible.\n\n"

            "6. Forces — A measure of how much residual force is acting on each atom after minimization, "
            "typically in eV/Angstrom. Ideally, after a successful minimization, forces on all atoms "
            "should be very close to zero, meaning every atom has settled into a true energy minimum. "
            "Large residual forces mean the structure did not fully relax — the atoms are still being "
            "pushed or pulled by their neighbors, which is a sign that the atomic arrangement was "
            "physically unrealistic or that the structure became trapped in a high-energy local minimum "
            "during relaxation rather than finding the true lowest-energy arrangement.\n\n"

            "7. Stress — A measure of the internal mechanical tension or compression remaining in the "
            "unit cell after minimization, typically reported in kilobars (kbar) or GPa. Ideally, after "
            "a successful minimization, stress should be close to zero, meaning the unit cell size and "
            "shape are also at their optimal values. Large residual stress means the unit cell dimensions "
            "are still wrong for the atomic arrangement — the atoms are being squeezed too tightly "
            "together or are too far apart — which is another sign of an unrealistic or trapped "
            "structure.\n\n"

            "=== HOW TO INTERPRET THE TEST RESULTS AND DECIDE WHAT TO CHANGE ===\n"
            "Your goal is to propose new PyXtal parameters that are more likely to produce a structure "
            "with a lower (more negative) energy, near-zero residual forces, and near-zero residual "
            "stress than the last structure achieved. Use the following chemistry and crystallography "
            "reasoning to guide your decision:\n\n"

            "Energy interpretation:\n"
            "- If the energy is very high (close to zero or positive), the last structure was likely "
            "very poor — badly distorted, with atoms too close together or in chemically unreasonable "
            "positions. This suggests the space group chosen imposed symmetry constraints that forced "
            "atoms into unfavorable positions, OR that NumIons produced a unit cell that was too crowded "
            "or too sparse. You should consider trying a significantly different space group, or adjusting "
            "Z (the number of formula units in the unit cell, which scales all of NumIons) to change the "
            "packing.\n"
            "- If the energy is moderately low but not as low as you would expect for this compound type, "
            "the structure may be in a local minimum — a somewhat stable arrangement but not the global "
            "best. Consider trying a closely related space group (e.g., a subgroup or supergroup of the "
            "current one) that allows slightly different atomic arrangements, which may let the structure "
            "escape toward a lower energy.\n"
            "- If the energy is already very low and close to what you would expect for this compound "
            "based on known similar materials, you may only need to make small refinements, such as "
            "trying a slightly different Z or a closely related space group to confirm you are near the "
            "true global minimum.\n\n"

            "Forces interpretation:\n"
            "- Large residual forces (e.g., above ~0.1 eV/Angstrom as a rough guideline) mean the "
            "minimization did not converge to a true energy minimum. This often indicates the initial "
            "structure from PyXtal was so geometrically unreasonable that the minimizer could not find a "
            "valid low-energy arrangement from it. A different space group that imposes different "
            "geometric constraints may produce a better starting structure that the minimizer can "
            "actually relax properly.\n"
            "- Near-zero forces combined with a high energy is a warning sign that the structure "
            "converged to a local minimum rather than the global minimum — it found a stable point, but "
            "not the best one. Explore different space groups or Z values.\n\n"

            "Stress interpretation:\n"
            "- Large residual stress means the unit cell volume or shape is wrong for the arrangement of "
            "atoms. This can happen when NumIons places too many or too few atoms in the unit cell "
            "(making the effective atomic density too high or too low), or when the space group imposes "
            "a cell shape (e.g., cubic, hexagonal, orthorhombic) that is incompatible with how these "
            "atoms naturally want to pack. If stress is large, consider whether a different space group "
            "with a different cell geometry or a different Z value would better match the natural packing "
            "density of this compound.\n\n"

            "General space group reasoning:\n"
            "Higher-symmetry space groups (higher numbers generally, but specifically those with many "
            "symmetry operations like cubic groups 195-230, hexagonal/trigonal groups, etc.) impose "
            "stronger constraints on where atoms can sit. This can be beneficial if the compound truly "
            "has high symmetry, but harmful if the real structure has lower symmetry — forcing artificial "
            "constraints can prevent the structure from adopting its natural geometry. If the last "
            "structure performed poorly, consider whether moving to a lower-symmetry space group in the "
            "same crystal system, or a different crystal system entirely, might allow a more natural "
            "atomic arrangement. Base this on your chemical knowledge of what kinds of structures are "
            "known for compounds with this element combination and stoichiometry.\n\n"

            "=== WHAT YOU ARE ALLOWED TO CHANGE ===\n"
            "You may suggest changes to any of the four PyXtal parameters:\n"
            "- Dim: In almost all cases this should stay at 3. Only suggest changing it if the energy "
            "results and compound type strongly suggest the material is layered (Dim=2) or chain-like "
            "(Dim=1).\n"
            "- Group: You may suggest any space group number valid for the Dim you choose. Base your "
            "choice on chemical reasoning about what space group is likely to produce a better structure "
            "for this compound, informed by the test results from the last iteration.\n"
            "- Species: This should NOT change unless you have identified a reason to believe the "
            "original formula was misinterpreted. The elements present in the compound are fixed by its "
            "chemistry. Keep Species the same as it was in the last structure.\n"
            "- NumIons: You may change this by choosing a different Z value (number of formula units per "
            "unit cell) and multiplying all base atom counts by the new Z. Always ensure that the new "
            "NumIons values are compatible with the Wyckoff position multiplicities available in the new "
            "Group you are proposing (as explained below).\n\n"

            "=== CRITICAL WYCKOFF CONSTRAINT (DO NOT VIOLATE THIS) ===\n"
            "Within each space group, atoms can only be placed at certain symmetry-allowed positions "
            "called Wyckoff positions, each of which has a fixed 'multiplicity' — a fixed minimum number "
            "of atoms that must be placed together at that site because the space group's symmetry "
            "operations automatically generate copies of every atom. For example, a particular Wyckoff "
            "position might have multiplicity 4, meaning if you place one atom there, the symmetry "
            "automatically creates 3 more copies, giving 4 atoms total at that site. PyXtal builds the "
            "structure by decomposing the NumIons count for each element into a sum of Wyckoff "
            "multiplicities. If a NumIons value cannot be formed by adding together any combination of "
            "the available Wyckoff multiplicities for the chosen space group, PyXtal cannot place those "
            "atoms and will crash. Before finalizing your answer, verify mentally that every value in "
            "your proposed NumIons list can be decomposed into a valid combination of Wyckoff "
            "multiplicities for your proposed Group. If not, adjust Z or choose a different Group.\n\n"

            "=== STEP-BY-STEP REASONING PROCESS ===\n"
            "Work through the following steps before giving your answer:\n\n"

            "STEP 1 — Assess the quality of the last structure.\n"
            "Look at the Energy, Forces, and Stress together. Was the last structure good (low energy, "
            "near-zero forces and stress), mediocre (moderate energy, some residual forces/stress), or "
            "poor (high energy, large forces and/or stress)? This tells you how much you need to change.\n\n"

            "STEP 2 — Identify the most likely reason the last structure was not optimal.\n"
            "Based on the test results and your knowledge of the compound's chemistry, reason about why "
            "the last space group and NumIons combination produced the result it did. Was the space group "
            "probably too high in symmetry? Too low? Did the unit cell likely have the wrong number of "
            "atoms (Z too large or too small)? Were atoms likely forced into geometrically unreasonable "
            "positions?\n\n"

            "STEP 3 — Decide on the new Space Group.\n"
            "Choose a new Group based on your reasoning from Step 2 and your knowledge of what space "
            "groups are commonly observed in real compounds with this element combination and "
            "stoichiometry. If the last result was very poor, consider a substantially different space "
            "group. If it was close to good, try a closely related one.\n\n"

            "STEP 4 — Determine the new Z and compute new NumIons.\n"
            "Choose a Z value appropriate for your new space group and the compound type. Multiply every "
            "base atom count (the smallest whole-number ratio from the chemical formula) by Z to get the "
            "new NumIons list. Ensure the order of NumIons still matches the order of Species exactly.\n\n"

            "STEP 5 — Verify Wyckoff compatibility.\n"
            "Confirm that every value in your new NumIons list can be decomposed into valid Wyckoff "
            "multiplicities for your new Group. If not, adjust Z or choose a different Group and repeat "
            "from Step 3.\n\n"

            "=== OUTPUT FORMAT (FOLLOW EXACTLY) ===\n"
            "Return ONLY the following four lines, and nothing else — no explanation, no markdown "
            "formatting, no code block symbols, no extra text before or after:\n\n"
            "Dim: <integer>\n"
            "Group: <integer>\n"
            "Species: <list of element symbol strings, e.g., ['Cu', 'C', 'O', 'H']>\n"
            "NumIons: <list of integers, e.g., [4, 2, 10, 4]>"




            ),

        },
        {
            "role": "user",
            "content": "Prompt: Determine the PyXtal Parameters That should be used for the next crystal generation based on the values of thease" 
            " tests from the last Crystal. Thease were the values of the last crystal: "
            f"Dim: {Dim}, Space Group: {Group}, Species: {Species}, NumIons: {NumIons} "
            "Bellow are the tests that were run on the last structure and its results: "
            f"Energy: {Energy}, Forces: {Forces}, Stress: {Stress}. This is this crystals formula: {formula}"
            "Figure out what the values for Dim, Space Group, Species, and NumIons should be for the next structure thats generated to create the most "
            "stable, and most accurate structure possible. "
        }
    ]

    response = client.chat_completion(
        model = "Qwen/Qwen2.5-72B-Instruct",
        messages = messages,
        max_tokens = 8000,
        temperature = 0.2,
    )

    return response.choices[0].message.content





#______________________________________________________________________________


###############################################################################

#______________________________________________________________________________

# Main function

if __name__ == '__main__':

    Beta_Array = []

    with open("BetaFile", "r", encoding = "utf-8") as file:
        Beta_Array = json.load(file)
 
    for entry in Beta_Array:
        

        print("\n")
        print("--------------------------------------------------- Start of Next Compound ---------------------------------------------")

        print("\n")

        InitalSuperGammaResult = InitalSuperGammaFunction(entry["Formula"]) #LLM gives inital values to use in PyXtal Structure Generation
        print(f"Formula: {entry["Formula"]}")
        #print(f"InitalSuperGammaFunction Result {InitalSuperGammaResult}")

        #__________________________________________________________________________________
        #Value Parsing

        print("\n")

        print("------------------------Value Parsing-----------------------------------")

        
        Dim = int(SubGammaFunction(InitalSuperGammaResult, "Dim:", "Group"))
        print(f"Dim:{Dim}")

        Group = int(SubGammaFunction(InitalSuperGammaResult, "Group:", "Species"))
        print(f"Group:{Group}")

        SpeciesSubGammaFunctionResult = SubGammaFunction(InitalSuperGammaResult, "Species:", "NumIons").replace("'", '"')
        print(f"SpeciesSubGammFunctionResult:{SpeciesSubGammaFunctionResult}")
        Species = json.loads(SpeciesSubGammaFunctionResult)
        print(f"Species:{Species}")

        NumIonsSubGammaFunctionResult = SubGammaFunction(InitalSuperGammaResult, "NumIons:", "end").replace("'", '"')
        print(f"NumIonsSubGammFunctionResult: {NumIonsSubGammaFunctionResult}")
        NumIons = json.loads(NumIonsSubGammaFunctionResult)
        print(f"NumIons:{NumIons}")
        

        #____________________________________________________________________________________

        #___________________________________________

        #Temp Values for structure gen so we dont have to use AI credits every test

        # Dim = 3
        # Group = 122
        # Species = ["N", "H", "P", "O"]
        # NumIons = [4, 24, 4, 16]
        


        #___________________________________________

        print("\n")
        # print("--------------------Crystal Structure generation---------------------------")
        GammaOneFunctionCrystalStructure = GammaOneFunction(Dim, Group, Species, NumIons)
        print(f"GammaOneFunction Crystal Strucutre Information {GammaOneFunctionCrystalStructure}")
        print(f"Formula: {GammaOneFunctionCrystalStructure.formula}")
        print(f"Atoms: {GammaOneFunctionCrystalStructure.numIons}")
        print(f"Space Groups: {GammaOneFunctionCrystalStructure.group}")

        GammaOneFunctionCrystalStructure.to_file(f"PyXtalStructure.cif")

        #______________________________________________________________________________________________

        print("\n")

        print("----------------------------------------- CHGNet Relaxation ----------------------------------------------------")

        print("\n")


        # --------------------- CHGNet Relaxation --------------------------
        GammaTwoFunctionResult = GammaTwoFunction()
        print(f"GammaTwoFunction result: {GammaTwoFunctionResult}")
        #__________________________________________________________________

        # Relaxation results printing and parsing

        Relaxed_Structure = GammaTwoFunctionResult["final_structure"]
        print(f"Relaxed Structure: {Relaxed_Structure}")
        Total_Energy = GammaTwoFunctionResult["total_energy"]
        print(f"Total Energy: {Total_Energy}")
        ForcesArray = GammaTwoFunctionResult["forces"]
        print(f"ForcesArray: {ForcesArray}")
        StressArrays = GammaTwoFunctionResult["stress"]
        print(f"StressArray: {StressArrays}")
        # Chech to see if we can get individual stress numbers print(f"stress[0][0]: {Stress[0][0]}")


        #________________________________________________________________

        StressTestPass = True


        # Stress Test
        for i in range (len(StressArrays)):
            SubStressArray = StressArrays[i]
            for j in range (len(SubStressArray)):
                print(f"Testing Stress of: {SubStressArray[j]}")
                print("\n")
                if abs(SubStressArray[j]) > 0.1:
                    StressTestPass = False
                    print(f" Stress Test failed for {SubStressArray[j]}")



        #_____________________________________________________________________


        endSecondaryLoop = True # Set to true right now so we can run onece and see output
        itteration = 0

        ForcesTestPass = True

        # Forces Test
        for i in range (len(ForcesArray)):
            SubForcesArray = ForcesArray[i]
            for j in range (len(SubForcesArray)):
                print(f"Testing Forces of: {SubForcesArray[j]}")
                print("\n")
                if abs(SubForcesArray[j]) > 0.1:
                    ForcesTestPass = False
                    print(f" Forces Test failed for {SubForcesArray[j]}")


        if StressTestPass == True and ForcesTestPass == True:
            export_structure_file = CifWriter(Relaxed_Structure, symprec=0.1)
            export_structure_file.write_file(f"relaxedStructure_{entry['Formula']}.cif")
            print("\n")
            print("\n")
            print(f"{entry['Formula']} passed and the structure was created")
            #Dont have enough credits to test and build this
        # else: # Secondary Loops that tries to come up with best results with this structure
        #     while not endSecondaryLoop or itteration < 10: #Want to run loop when ending is set to false / no / dont end OR if we have gon through this loop over 9 times and we are getting nowhere
        #         secondarySuperGammaFunctionResult = secondarySuperGammaFunction(Total_Energy, ForcesArray, StressArrays, Dim, Group, Species, NumIons, entry['Formula'] )
        #         print (secondarySuperGammaFunctionResult)