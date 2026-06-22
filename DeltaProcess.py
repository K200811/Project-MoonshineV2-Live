from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element
from pymatgen.analysis.local_env import CrystalNN

import math

import torch

import warnings


import json
import sys




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
    nn_finder = CrystalNN()

    for index, site in enumerate(structure):
        cn = nn_finder.get_cn(structure, index)
        print(f"Calculated CN {cn} for site {index}")

        try:
            expected_charge = abs(float(site.specie.oxi_state))
        except AttributeError:
            print(f"Error: Site {index} ({site.species_string}) does not have an oxidation state assigned.")
            return False

        # Calculate the Bond Valence Sum (BVS) for this site using CrystalNN neighbors
        neighbors = nn_finder.get_nn_info(structure, index)
        site_bvs = 0.0
        
        for nb in neighbors:
            nb_site = nb['site']
            
            # Universal BVS Parameter Equation fallback calculation:
            # R0 is approximately the sum of the atomic/covalent radii
            r_cation = site.specie.element.atomic_radius or site.specie.element.covalent_radius
            r_anion = nb_site.specie.element.atomic_radius or nb_site.specie.element.covalent_radius
            
            if r_cation is None or r_anion is None:
                continue
                
            r0 = r_cation + r_anion
            b = 0.37  # Universal constant parameter for typical inorganic bonds
            
            bond_length = structure.get_distance(index, nb['site_index'])
            
            # Classic Brown & Altermatt equation: v = exp((R0 - d) / b)
            vij = math.exp((r0 - bond_length) / b)
            site_bvs += vij

        # Check the error difference between calculated BVS and the assigned state
        bvsError = abs(site_bvs - expected_charge)

        # Allow a 0.35 cushion for machine-learning-relaxed (CHGNet) coordinate variances
        if bvsError <= 0.35:  
            print(f"Coordination Number test passed for site {site.species_string} -- Expected: {expected_charge}, Calculated BVS: {site_bvs:.2f}")
        else:
            print(f"❌ Test Failed for site {site.species_string} -- Expected: {expected_charge}, Calculated BVS: {site_bvs:.2f} (Error: {bvsError:.2f})")
            return False

    print("Total Coordination Numbers test passed")
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
        OverlapCheckResults = OverlapCheck(f"relaxedStructure_{formula}.cif", HasHydrogen)

        if OverlapCheckResults == True:
            print("In main -- Overlap Detected")
            sys.exit(1)
        if OverlapCheckResults == False:
            print("In Main -- There is no overlap")
        
        # ------------------------------------------------------- Is All Metalic Check ---------------------------------------------------------------

        print("\n")
        print("-------------------- RUNNING IS ALL METAL CHECK ----------------------")

        IsFulleymetalicResult = isFullyMetalic(f"relaxedStructure_{formula}.cif")

        if IsFulleymetalicResult == True:
            #-----------Run Composition based Check------------ 
            print("------------ Composition Based Oxidization test -----------")
            CompositionBasedOxidizationTestResult = CompositionBasedOxidizationTest(f"relaxedStructure_{formula}.cif", formula)
            if CompositionBasedOxidizationTestResult == False:
                sys.exit(1)
            # All printing for this function takes place in the function

        else: # Run a bond based check
            print("\n")
            print("--------------- Bond Based Oxidization test ----------------")
            BondBasedOxidizationTestResult = BondBasedOxidizationTest(f"relaxedStructure_{formula}.cif", formula )
            if(BondBasedOxidizationTestResult == False):
                sys.exit(1)
            # All printing takes place in the function

        # -------------------------------- Run Coordination Numbers Test --------------------
        
        print("\n")
        print("---------------------- RUNNING COORDINATION NUMBERS TEST -----------------------")
        CoordinationNumbersTestResult = CoordinationNumbersTest(f"{formula}_Oxidization_Assigned_Structure.cif")
        if CoordinationNumbersTestResult == False:
            sys.exit(1)
        # All printing takes place in the function