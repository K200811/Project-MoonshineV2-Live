import logging
import json
import time
import math


import re
from typing import List, Optional, Sequence, Tuple

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
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from ase.filters import FrechetCellFilter
from pyxtal.symmetry import Group
from pymatgen.core import Composition

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter

start_time = time.perf_counter()

with open("moonshine_data.json", "r") as f:
    data = json.load(f)

data["payload"]["current_stage"] = "GammaProcess"

with open("moonshine_data.json", "w") as f:
    json.dump(data, f, indent=2)

NUMBER_OF_ITTERATIONS = 50

# ──────────────────────────────────────────────────────────────────────────────
# Initial Super Gamma Function  (generates PyXtal parameters)
# ──────────────────────────────────────────────────────────────────────────────

def InitalSuperGammaFunction(
    sg,
    formula: str,
    species: Sequence[str],
    max_total: int = 40,
    max_formula_units: Optional[int] = None,
    ) -> Optional[List[int]]:

    # --- parse formula into element -> integer count ---
    pattern = r"([A-Z][a-z]?)(\d*\.?\d*)" 
    counts = {}
    for el, num in re.findall(pattern, formula): 
        if not el:
            continue
        n = float(num) if num else 1.0
        counts[el] = counts.get(el, 0.0) + n

    missing = [el for el in counts if el not in species]
    if missing:
        raise ValueError(f"species list is missing elements found in formula: {missing}")
    extra = [el for el in species if el not in counts]
    if extra:
        raise ValueError(f"species list has elements not present in formula: {extra}")

    # all parsed counts should be (near) integers
    for el, n in counts.items():
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"non-integer subscript for {el} in formula '{formula}': {n}")

    ratio = [int(round(counts[el])) for el in species]

    # reduce to the simplest integer ratio (does NOT alter the formula
    # parsing above, only how Z is scaled from here)
    g = math.gcd(*ratio) if len(ratio) > 1 else ratio[0]
    g = g or 1
    r = [x // g for x in ratio]

    # --- search over Z = 1, 2, 3, ... ---
    group = Group(sg)

    z = 1
    while True:
        numIons = [ri * z for ri in r]
        if sum(numIons) > max_total:
            break
        if max_formula_units is not None and z > max_formula_units:
            break
        compatible, _ = group.check_compatible(numIons)
        if compatible:
            return numIons
        z += 1

    return None


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
# PyXtal structure generation
# ──────────────────────────────────────────────────────────────────────────────

def GammaOneFunction(dim, group, species, numIons):
    Crystal = pyxtal()
    Crystal.from_random(
        dim=dim,
        group=group,
        species=species,
        numIons=numIons,
    )
    return Crystal


# ──────────────────────────────────────────────────────────────────────────────
# CHGNet relaxation — coarse pass (UnitCellFilter + FIRE)
# ──────────────────────────────────────────────────────────────────────────────

def GammaTwoFunction(file):
    print(f"CUDA availability check: {torch.cuda.is_available()}")

    structure = Structure.from_file(file)

    adapter = AseAtomsAdaptor()
    ase_atoms = adapter.get_atoms(structure)

    chgnet_model = CHGNet.load()
    calculator = CHGNetCalculator(model=chgnet_model, use_device="cpu")
    ase_atoms.calc = calculator

    symmetry_constraint = FixSymmetry(ase_atoms)
    ase_atoms.set_constraint(symmetry_constraint)

    ecf = UnitCellFilter(ase_atoms)

    optimizer = FIRE(ecf, logfile=None)
    optimizer.run(fmax=0.01, steps=3000)

    final_structure = adapter.get_structure(ase_atoms)
    crystal_info = ase_atoms.calc.results

    analyser = SpacegroupAnalyzer(final_structure, symprec=0.2)
    print(f"Detected Space Group: {analyser.get_space_group_symbol()}")

    correct_structure = analyser.get_symmetrized_structure()

    result = {
        "final_structure": correct_structure,
        "total_energy": crystal_info["energy"],
        "forces": crystal_info["forces"],
        "stress": crystal_info["stress"],
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CHGNet relaxation — fine pass (FrechetCellFilter + LBFGS)
# ──────────────────────────────────────────────────────────────────────────────

def GammaThreeFunction(file):
    print(f"CUDA availability check: {torch.cuda.is_available()}")

    structure = Structure.from_file(file)

    adapter = AseAtomsAdaptor()
    ase_atoms = adapter.get_atoms(structure)

    chgnet_model = CHGNet.load()
    calculator = CHGNetCalculator(model=chgnet_model)
    ase_atoms.calc = calculator

    symmetry_constraint = FixSymmetry(ase_atoms)
    ase_atoms.set_constraint(symmetry_constraint)

    ecf = FrechetCellFilter(ase_atoms)

    optimizer = LBFGS(ecf, logfile=None)
    optimizer.run(fmax=0.001, steps=7000)

    final_structure = adapter.get_structure(ase_atoms)
    crystal_info = ase_atoms.calc.results

    analyser = SpacegroupAnalyzer(final_structure, symprec=0.001)
    print(f"Detected Space Group: {analyser.get_space_group_symbol()}")

    correct_structure = analyser.get_symmetrized_structure()

    result = {
        "final_structure": correct_structure,
        "total_energy": crystal_info["energy"],
        "forces": crystal_info["forces"],
        "stress": crystal_info["stress"],
    }
    return result

def StressTest (stress_arrays):

    for row in stress_arrays:
        for val in row:
            print(f"Testing Stress of: {val}\n")
            if abs(val) > 0.1:
                stress_pass = False
                print(f"Stress Test failed for {val}")
                data["payload"]["logs"]["warnings"].append(f"Stress Test failed for {val}")
                return False
    return True


def ForcesTest(forces_array):

    forces_pass = True
    for row in forces_array:
        for val in row:
            print(f"Testing Forces of: {val}\n")
            if abs(val) > 0.1:
                forces_pass = False
                print(f"Forces Test failed for {val}")
                data["payload"]["logs"]["warnings"].append(f"Forces Test failed for {val}")
                return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    Beta_Array = []
    GammaArray = []

    with open("BetaFile", "r", encoding="utf-8") as file:
        Beta_Array = json.load(file)

    for entry in Beta_Array:

        print("\n")
        print("--------------------------------------------------- Start of Next Compound ---------------------------------------------\n")

        try:
            composition = Composition(entry["Formula"])
            species = [el.symbol for el in composition.elements]
        except Exception as e:
            print(f" Faield to parse the composition for {entry['Formula']}: {e}")
            continue

        for SPACEGROUP in range(229):

            for ITTERATIONS in range (NUMBER_OF_ITTERATIONS):


                SPACEGROUP = SPACEGROUP + 1
                print(f"--------------------------------------------- TESTING SPACE GROUP {SPACEGROUP} -----------------------------")
                
                try: 
                    numions  = InitalSuperGammaFunction(SPACEGROUP ,entry["Formula"], species, 100) 
                except Exception as e:
                    print(f"InitalSuperGammaFunction failed for {entry['Formula']} SG {SPACEGROUP}: {e}")
                    continue

                if numions is None:
                    print(f"No valid numIons found for {entry['Formula']} in SG {SPACEGROUP} — skipping")
                    continue

                print(f"Formula: {entry['Formula']}")

                print("\n------------------------Value Parsing-----------------------------------\n")

                Dim = 3 # Only running Dim 3 for this verion
                print(f"Dim:{Dim}")

                SpaceGroup = SPACEGROUP 
                print(f"Group:{Group}")

                Species = species
                print(f"Species:{Species}")

                NumIons = numions
                print(f"NumIons:{NumIons}")
                # ─────────────────────────────────────────────────────────────────────



                # ----------------------------- PyXtal Structure Generation -------------------

                print("\n--------------------Crystal Structure Generation---------------------------")

                try:
                    crystal = GammaOneFunction(Dim, SpaceGroup, Species, NumIons)
                except Exception as e:
                    print(f"GammaOneFunction failed for {entry['Formula']} SG {SPACEGROUP}: {e}")
                    continue

                print(f"GammaOneFunction Crystal Structure Information: {crystal}")
                print(f"Formula: {crystal.formula}")
                print(f"Atoms: {crystal.numIons}")
                print(f"Space Group: {crystal.group}")
        
                try:
                    crystal.to_file(f"PyXtalStructure_{entry['Formula']}_{ITTERATIONS}.cif")
                except Exception as e:
                    print(f"Failed to write CIF for {entry['Formula']} SG {SPACEGROUP}: {e}")
                    continue

                # ---------------------------- Inital CHGNet Relaxation ----------------------------
                print("\n----------------------------------------- INITAL CHGNET RELAXATION ----------------------------------------------------\n")
                
                try:
                    GammaTwoResult = GammaTwoFunction(f"PyXtalStructure_{entry['Formula']}_{ITTERATIONS}.cif")
                except Exception as e:
                    print(f"GammaTwoFunction failed for {entry['Formula']} SG {SPACEGROUP}: {e}")
                    continue
                print(f"GammaTwoFunction result: {GammaTwoResult}")

                Relaxed_Structure = GammaTwoResult["final_structure"]
                Total_Energy      = GammaTwoResult["total_energy"]
                ForcesArray       = GammaTwoResult["forces"]
                StressArrays      = GammaTwoResult["stress"]

                print(f"Relaxed Structure: {Relaxed_Structure}")
                print(f"Total Energy: {Total_Energy}")
                print(f"ForcesArray: {ForcesArray}")
                print(f"StressArray: {StressArrays}")

                StressTestPass = StressTest(StressArrays)
                ForcesTestPass = ForcesTest(ForcesArray)

                RunGammaThree = False
                if StressTestPass and ForcesTestPass and (Total_Energy / len(Relaxed_Structure)) < 0:
                    export = CifWriter(Relaxed_Structure, symprec=0.1)
                    export.write_file(f"relaxedStructure_{entry['Formula']}_{ITTERATIONS}.cif")
                    print(f"Ev/atom: {Total_Energy / len(Relaxed_Structure)}\n")
                    print(f"{entry['Formula']} passed the coarse relaxation — structure written.\n")
                    GammaArray.append({
                        "Formula": entry["Formula"],
                        "GammaTwoFunction Stress Array": StressArrays.tolist() if hasattr(StressArrays, "tolist") else StressArrays,
                        "GammaTwoFunction Forces Array": ForcesArray.tolist() if hasattr(ForcesArray, "tolist") else ForcesArray,
                    })
                    RunGammaThree = True
                else:
                    print("Stress Test failed Gamma three")
                    continue

                # --------------------------------- Secondary CHGNet Relaxation ---------------------

                if RunGammaThree:
                    print("\n------------------------- Running Gamma Three ---------------------")
                    try:
                        GammaThreeResult = GammaThreeFunction(f"relaxedStructure_{entry['Formula']}_{ITTERATIONS}.cif")
                    except Exception as e:
                        print(f"GammaThreeFunction failed for {entry['Formula']} SG {SPACEGROUP}: {e}")
                        continue
                    print(f"GammaThreeFunction result: {GammaThreeResult}")

                    Final_Relaxed_Structure = GammaThreeResult["final_structure"]
                    Total_Energy            = GammaThreeResult["total_energy"]
                    ForcesArray             = GammaThreeResult["forces"]
                    StressArrays            = GammaThreeResult["stress"]

                    print(f"Relaxed Structure: {Final_Relaxed_Structure}")
                    print(f"Total Energy: {Total_Energy}")
                    print(f"ForcesArray: {ForcesArray}")
                    print(f"StressArray: {StressArrays}")

                    StressTestPass = StressTest(StressArrays)
                    ForcesTestPass = ForcesTest(ForcesArray)

                    if StressTestPass and ForcesTestPass:
                        export = CifWriter(Final_Relaxed_Structure, symprec=0.1)
                        export.write_file(f"Final_relaxedStructure_{entry['Formula']}_{ITTERATIONS}.cif")
                        print(f"\n{entry['Formula']} passed the fine relaxation — final structure written.\n")
                        GammaArray.append({
                            "Formula": entry["Formula"],
                            "GammaThreeFunction Stress Array": StressArrays.tolist() if hasattr(StressArrays, "tolist") else StressArrays,
                            "GammaThreeFunction Forces Array": ForcesArray.tolist() if hasattr(ForcesArray, "tolist") else ForcesArray,
                        })
                    else:
                        print("Stress Test Failed for Gamma Three")
                        continue

    # ------------------- End data collection -------------------
    end_time       = time.perf_counter()
    execution_time = end_time - start_time

    data["payload"]["stage_timing"][2]["seconds"] = execution_time

    # Fix: initialise as list before appending (was "" in original)
    data["payload"]["candidates_in_system"] = []
    for i, entry in enumerate(GammaArray):
        data["payload"]["candidates_in_system"].append({
            "formula": entry["Formula"],
            "index": i,
            "id": f"cand_{i}",
            "status": "Passed Gamma",
        })

    with open("moonshine_data.json", "w") as f:
        json.dump(data, f, indent=2)

    with open("GammaFile", "w") as json_file:
        json.dump(GammaArray, json_file, indent=4)