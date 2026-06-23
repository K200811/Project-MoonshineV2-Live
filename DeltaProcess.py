from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments


import math

import torch

import warnings


import json
import sys


CoordinationNumberLimitsPerElement = {
# --- Alkali Metals ---
    "H":  (1, 2),    # Terminals or bridging hydrides
    "Li": (3, 8),    # Usually 4 or 6, up to 8 in dense oxides/halides
    "Na": (4, 9),    # Commonly 6, stretches to 8 or 9 in complex frameworks
    "K":  (6, 12),   # Large ionic radius, highly coordinated
    "Rb": (6, 12),
    "Cs": (8, 12),   # CsCl structure has CN=8; perovskites have CN=12

    # --- Alkaline Earth Metals ---
    "Be": (3, 4),    # Highly rigid, strictly tetrahedral or occasionally trigonal planar
    "Mg": (4, 8),    # Dominated by octahedral (6), 4 in spinels, 8 in some silicates
    "Ca": (6, 12),   # Typically 6 to 8, 12 in perovskite A-sites
    "Sr": (6, 12),
    "Ba": (6, 12),

    # --- Metalloids & Nonmetals (P-Block) ---
    "B":  (3, 6),    # Planar BO3 (3) or tetrahedral BO4 (4); 6 in borides
    "C":  (2, 4),    # Linear, trigonal planar, or tetrahedral
    "N":  (1, 4),    # Nitrides are typically low coordination
    "O":  (2, 6),    # 2 (bridging), 3/4 (standard oxides), 6 (rock salt MgO/CaO)
    "F":  (1, 4),    # Similar to Oxygen but slightly more restrictive
    "Al": (4, 6),    # Rigid tetrahedra or octahedra
    "Si": (4, 6),    # Highly rigid SiO4 (4) or high-pressure stishovite (6)
    "P":  (3, 6),    # Phosphates (4), phosphides (up to 6)
    "S":  (2, 6),    # Sulfides usually mirror oxides but with larger cation accommodations
    "Cl": (1, 6),    # Rock-salt alkali halides have CN=6
    "Ga": (4, 6),
    "Ge": (4, 6),
    "As": (3, 6),
    "Se": (2, 6),
    "Br": (1, 6),
    "In": (4, 8),
    "Sn": (4, 8),
    "Sb": (3, 6),
    "Te": (2, 6),
    "I":  (1, 6),
    "Tl": (6, 12),
    "Pb": (4, 12),   # Can have highly lone-pair distorted 4, up to 12 in perovskites
    "Bi": (3, 9),

    # --- Early Transition Metals ---
    "Sc": (6, 9),
    "Ti": (4, 8),    # Rutile/Anatase are strictly 6, some Ti-complexes are 4 or 5
    "V":  (4, 8),
    "Cr": (4, 6),    # Rigid octahedral Cr(III), tetrahedral Cr(VI)
    "Mn": (4, 8),    # Variable oxidation states yield highly flexible packing
    "Fe": (4, 8),    # 4 (spinels), 6 (common oxides/sulfides), 8 (garnets)
    "Co": (4, 6),
    "Ni": (4, 6),

    # --- Late & Heavy Transition Metals ---
    "Y":  (6, 9),
    "Zr": (6, 8),    # Zirconia (ZrO2) displays 7 or 8 coordination
    "Nb": (4, 8),
    "Mo": (4, 8),    # MoS2 layer types show trigonal prismatic (6)
    "Tc": (4, 6),
    "Ru": (4, 6),
    "Rh": (4, 6),
    "Pd": (4, 6),    # Square planar (4) or octahedral (6)
    "Ag": (2, 6),    # Can form linear coordination (2) like Ag(I)
    "Cd": (4, 8),
    "Hf": (6, 8),
    "Ta": (4, 8),
    "W":  (4, 8),
    "Re": (4, 6),
    "Os": (4, 6),
    "Ir": (4, 6),
    "Pt": (4, 6),
    "Au": (2, 6),    # Linear 2-coordination is highly common for gold
    "Hg": (2, 6),    # Linear Hg(II) in cinnabar/oxides

    # --- Coinage / Post-Transition Metals ---
    "Cu": (2, 6),    # Linear Cu(I) (2), square-planar/pyramidal Cu(II) (4-5), octahedral (6)
    "Zn": (4, 6),    # Strictly tetrahedral in wurtzite/zincblende, octahedral in oxides

    # --- Lanthanides (Rare Earths: Highly Coordinated) ---
    "La": (6, 12),   # High coordination numbers due to massive atomic radii
    "Ce": (6, 12),
    "Pr": (6, 12),
    "Nd": (6, 12),
    "Pm": (6, 12),
    "Sm": (6, 12),
    "Eu": (6, 12),
    "Gd": (6, 12),
    "Tb": (6, 12),
    "Dy": (6, 12),
    "Ho": (6, 12),
    "Er": (6, 12),
    "Tm": (6, 12),
    "Yb": (6, 12),
    "Lu": (6, 12),

    # --- Actinides ---
    "Th": (6, 12),
    "Pa": (6, 12),
    "U":  (6, 12),   # Uranyl ions display equatorial coordination shells (6 to 8)
    "Np": (6, 12),
    "Pu": (6, 12),
    "Am": (6, 12),
    
    # --- Intermetallics / Pure Elements Exception Catch-All ---
    # For bulk pure metals (like FCC, BCC, HCP) or intermetallic alloys, 
    # coordination numbers naturally range from 8 to 12 (or 14-16 for cluster cages). 
    # If screening an all-metal system, adjust boundaries using the helper method below.
}

# Checks to make sure that no atoms are overlaping

def OverlapCheck (file: str, hasHydrogen: bool):
    structure = Structure.from_file(file)

    if hasHydrogen:
        cutoff = 0.6
    else:
        cutoff = 1.0

    allNeighbors = structure.get_all_neighbors(r= cutoff, include_image=True)

    overlapDetected = False #Set to false first becuase blue sheep is if there is a overlap

    for i, neighbors in enumerate(allNeighbors):
        if (len(neighbors) > 0):
            for neighborSingular in neighbors:
                print(f" In function -- There is a Overlap at Site {i} and overlaps with {neighborSingular.index} with a distance of {neighborSingular.nn_distance:.3f}")
                overlapDetected = True
                
        
    if not overlapDetected:
        print("In funtion -- No overlap detected ( BOYEA )")

    return overlapDetected

#___________________________________________________________________________

###########################################################################

#________________________________________________________________________

# Checks if the entire crystal is fully metalic or not to determin which oxidization test to use

def isFullyMetalic (file: str):

    structure = Structure.from_file(file)

    uniqeElements = structure.composition.elements

    allMetals = all(element.is_metal for element in uniqeElements)

    if allMetals:
        print("Is fulley metalic")
        return True
    else:
        print("is not fully metalic")
        return False
    
#_______________________________________________________________________

#########################################################################

#_______________________________________________________________________

# Gusses the oxidization of the crystal by composition of it

def CompositionBasedOxidizationTest (file: str,formula):

    structure = Structure.from_file(file)
    composition = structure.composition

    GussedOxidizationStates = composition.oxi_state_guesses()

    if GussedOxidizationStates: #If the list is emptey then nothing was found so we write this line to check of that list even exists
        print(f"{len(GussedOxidizationStates)} Valid Oxidization State(s) found, best guess is: {GussedOxidizationStates[0]}")
        print(f"Creating structure file with oxidization guesses as {formula}_Oxidization_Assigned_Structure")
        Charged_Structure = structure.add_oxidation_state_by_guess()
        Charged_Structure.to(filename= f"{formula}_Oxidization_Assigned_Structure.cif")

        return True
    else:
        print(f" No valid Oxidization found for {composition.reduced_formula}")
        return False
    
#___________________________________________________________________

####################################################################

#___________________________________________________________________

# Guess the oxidization states of the 

def BondBasedOxidizationTest (file: str, formula):

    structure = Structure.from_file(file)

    bva = BVAnalyzer()

    try:
        charged_structure = bva.get_oxi_state_decorated_structure(structure)
        total_charge = sum(site.specie.oxi_state for site in charged_structure)

        # Checks if the oxidization states assigned are total garbage

        for site in charged_structure:
            element = Element(site.specie.symbol)
            oxidizationState = site.specie.oxi_state

            if oxidizationState not in element.common_oxidation_states:
                print(f"{element.symbol} has invalid oxidization state of {oxidizationState}")

                return False
            
            
        for site in charged_structure:
            print(site.specie.symbol, site.specie.oxi_state)


        if abs(total_charge) < 1e-4:
            print("Valid Bond-Valance Oxidization states found")
            print(structure.composition)
            print(f"Saving Oxidazation attached structure as {formula}_Oxidization_Assigned_Structure")
            charged_structure.to(filename=f"{formula}_Oxidization_Assigned_Structure.cif")
            return True
        else:
            print(f"Could not find valid oxidization states the total charge is {total_charge:.2f}")
            print(structure.composition)
            return False
    except ValueError as e:
        print(f"Bond Lengths do not create a valid bond network that oxidization states can be assigned to Error: {e}")
        print(" Runnign composition based oxidization")
        CompositionBasedOxidizationTestResultInBondBasedFunction = CompositionBasedOxidizationTest(file, formula)
        return CompositionBasedOxidizationTestResultInBondBasedFunction

#__________________________________________________________________

#Coordination Numbers check that makes sure we are bonded to appropate number of atoms

def CoordinationNumbersTest (file): #Make sure the final input is the oxidized structure
    
    structure = Structure.from_file(file)
    cnn = CrystalNN(porous_adjustment=False)

    for i, site in enumerate(structure):
        element = site.specie.element.symbol

        try:
            cn = cnn.get_cn(structure, i)
        except Exception as e:
            print(f"Following error took place whiile getting the Coordination Number {e}")
            return False
        
        min_cn, max_cn = CoordinationNumberLimitsPerElement[element]

        if(min_cn <= cn <= max_cn):
            print(f"Coordination number for {element} is between the limits of {min_cn} nad {max_cn}. Coordination Number = {cn}")
        else:
            print(f"Coordination number of {element} is not between the limits of {min_cn} and {max_cn}. Coordination Number = {cn}")
            return False
    print("Full coordination numbers check passed")
    return True



def GeometryCheck (file, max_allowed_csm):
    structure = Structure.from_file(file)

    lgf = LocalGeometryFinder() # creates vorroni polyheara netwrok around every atom
    lgf.setup_structure(structure) #centers the calcualtions on our crystal structure

    se = lgf.compute_structure_environments() # calculates all the possibilites of the gemontries that are structure could be trying to be
    stratagy = SimplestChemenvStrategy() # setting up stratagy object
    lse = LightStructureEnvironments.from_structure_environments(
        strategy=stratagy,
        structure_environments=se
    ) # finds what is the most likely geometry this is trying to be

    for i, site in enumerate(structure):
        site_env = lse.coordination_environments[i]
        if not site_env: # if there is a error in getting this site (atom basicly) geometry
            continue
    
        if site_env is None or len(site_env) == 0:
            print(f"Warning: Could not determine site environment for site {i}. Skipping it.")
            continue
        else:
            csm = site_env[0]['csm']

        if csm > max_allowed_csm:
            print(f"Failed Geometry check, site {site.species_string} at index {i}, has a impossible geometry leading to a csm of {csm}")
            return False
        else:
            print(f"{site.species_string} passed local geometry check")
        
    print("Full Geometry Test passed")
    return True

            




# ----------------------------------------------------------------- Main Function ----------------------------------------------------------- #

if __name__ == "__main__":

    GammaArray = []

    with open("GammaFile", "r", encoding = "utf-8") as file:
        GammaArray = json.load(file)

    for entry in GammaArray:

        print("\n")
        print("-------------------------------------------------------NEW COMPOUND-----------------------------------------------------------------------")

        HasHydrogen = False
        formula = entry
        print(f"FORMULA: {formula}")
        # ----------------------------------------------- Overlap Check ---------------------------------------------------------------------
        if "H" in formula:
            HasHydrogen = True

        print("\n")
        print("--------------- RUNNING OVERLAPING CHECK  ------------------------")
        OverlapCheckResults = OverlapCheck(f"Final_relaxedStructure_{formula}.cif", HasHydrogen)

        if OverlapCheckResults == True:
            print("In main -- Overlap Detected")
            continue
        if OverlapCheckResults == False:
            print("In Main -- There is no overlap")
        
        # ------------------------------------------------------- Is All Metalic Check ---------------------------------------------------------------

        print("\n")
        print("-------------------- RUNNING IS ALL METAL CHECK ----------------------")

        IsFulleymetalicResult = isFullyMetalic(f"Final_relaxedStructure_{formula}.cif")

        if IsFulleymetalicResult == True:
            #-----------Run Composition based Check------------ 
            print("------------ Composition Based Oxidization test -----------")
            CompositionBasedOxidizationTestResult = CompositionBasedOxidizationTest(f"Final_relaxedStructure_{formula}.cif", formula)
            if CompositionBasedOxidizationTestResult == False:
                continue
            # All printing for this function takes place in the function

        else: # Run a bond based check
            print("\n")
            print("--------------- Bond Based Oxidization test ----------------")
            BondBasedOxidizationTestResult = BondBasedOxidizationTest(f"Final_relaxedStructure_{formula}.cif", formula )
            if(BondBasedOxidizationTestResult == False):
                continue
            # All printing takes place in the function

        # -------------------------------- Run Coordination Numbers Test --------------------
        
        print("\n")
        print("---------------------- RUNNING COORDINATION NUMBERS TEST -----------------------")
        CoordinationNumbersTestResult = CoordinationNumbersTest(f"{formula}_Oxidization_Assigned_Structure.cif")
        if CoordinationNumbersTestResult == False:
            continue
        # All printing takes place in the function

        print("\n")
        print("---------------------------------- RUNNING GEOMETRY CHECK ---------------------------")
        GeometryCheckResult = GeometryCheck(f"{formula}_Oxidization_Assigned_Structure.cif", 4.0)
        if GeometryCheckResult == False:
            continue
        #all printing takes place in the function
