from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.ext.matproj import MPRester

import json


MP_API_KEY = ""

def GetDesity (file):
    structure = Structure.from_file(file)
    density = structure.density

    if density > 0:
        return density
    else: 
        return ("Negitve Desity")
    
#__________________________________________________________________________
    
def getVolume(file):
    structure = Structure.from_file(file)

    volume = structure.volume

    if(volume > 0):
        return volume
    else:
        return ("Negitive Volume")
    
#_____________________________________________________________________________


def getLatticeParameters (file):
    structure = Structure.from_file(file)

    lattice = structure.lattice

    LengthXAxis = lattice.a
    LengthYAxis = lattice.b
    LengthZAxis = lattice.c

    angleAlpha = lattice.alpha
    angleBeta = lattice.beta
    angleGamma = lattice.gamma

    result = {
        "LengthXAxis Å ": LengthXAxis,
        "LengthYAxis Å ": LengthYAxis,
        "LengthZAxis Å ": LengthZAxis,

        "Alpha Angle ° ": angleAlpha,
        "Beta Angle ° ": angleBeta,
        "Gamma Angle ° ": angleGamma,
    }

    return result

#_________________________________________________________________________________

def getComposition (file):
    structure = Structure.from_file(file)

    composition = structure.composition

    reduced_formula = composition.reduced_formula
    Total_Atoms_In_unit_Cell = composition.formula

    element_dictonary = composition.as_dict()

    result = {
        "Reduced Formula": reduced_formula,
        "Total Atoms In The Unit Cell": Total_Atoms_In_unit_Cell,
        "Element Counts": element_dictonary
    }

    return result

#________________________________________________________________________________

def getSpaceGroup (file):

    structure = Structure.from_file(file)
    spaceGroup = structure.get_space_group_info()[1]
    
    return spaceGroup

#________________________________________________________________________________

def getLocalGeometry (file):

    structure = Structure.from_file(file)

    resultArray = []

    cnn = CrystalNN(porous_adjustment=True)

    motifs = ["tet", "oct", "sq_plan"]
    lsop = LocalStructOrderParams(motifs)

    for i, site in enumerate(structure):
        cn = cnn.get_cn(structure, i)

        nn_info = cnn.get_nn_info(structure, i)
        neighbor_indices = [info['site_index'] for info in nn_info]

        all_indices = [i] + neighbor_indices

        scores = lsop.get_order_parameters(structure, i, indices_neighs= neighbor_indices)

        results = dict(zip(motifs, scores))

        best_motif = max(results, key=results.get)
        best_score = results[best_motif]
        motif_names = {"tet": "Tetrahedral", "oct": "Octhedral", "sq_plan": "Square Planar"}
        geometry = motif_names.get(best_motif, "Unkown") if best_score > 0.5 else "Distorted/Unknown"
        
        result = {
            "Atom": site.species_string,
            "CN": cn,
            "Geometry": geometry
        }
        resultArray.append(result)

    return resultArray

#____________________________________________________________________

def GetbondLengths (file):
    resultArray = []
    structure = Structure.from_file(file)

    cnn = CrystalNN()

    for i, site in enumerate(structure): #Keeps track of index and item at index
        nn_info = cnn.get_nn_info(structure, i)
        for nn in nn_info:
            neighbor_site = nn['site']
            neighbor_index = nn['site_index']

            if i < neighbor_index:
                bond_length = structure.get_distance(i, neighbor_index)
                result = {
                    "BondLength": float(bond_length),
                    "SiteA": site.species_string,
                    "SiteB": neighbor_site.species_string 
                }
                resultArray.append(result)
    return resultArray

#_________________________________________________________________________________________

def prototypeMatching (file):

    structure = Structure.from_file(file)

    anonomusFormula = structure.composition.anonymized_formula
    spaceGroupNumber = structure.get_space_group_info()[1]

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True
    )

    matches = []

    with MPRester(MP_API_KEY) as mpr:

        docs = mpr.summary_search(
            formula = anonomusFormula,
            spacegroup_number =spaceGroupNumber,
            _fields= ["structure", "material_id", "formula_pretty"]
        )

        print(f"Found {len(docs)} candidates")

        for doc in docs:
            db_structure = doc["structure"]
            db_id = doc.get("task_id", doc.get("material_id", doc.get("id", "Unknown ID")))
            db_formula = doc.get("formula_pretty", "")

            if matcher.fit_anonymous(structure, db_structure):
                rms_distance = matcher.get_rms_dist(structure, db_structure) # How much you have to warp or distort your local structure to perfectly match the one in the database
                
                if rms_distance:
                    score = rms_distance[0]
                    matches.append({
                        "mp_id": db_id,
                        "prototype_formula": db_formula,
                        "rms_score": score
                    })

    if not matches:
        print("No prototype matches")
        return ("No Matches")
    matches = sorted(matches, key=lambda x: x["rms_score"])

    for rank, match in enumerate(matches, 1):
        print(f"Rank {rank}: {match['mp_id']} ({match['prototype_formula']}-type) | RMS Distortion: {match['rms_score']:.4f} Å")
    return matches[0]



#___________________________________________________________________________________________






if __name__ == '__main__':

    DeltaArray = []
    with open("DeltaFile", "r", encoding = "utf-8") as file:
        DeltaArray = json.load(file)

    for entry in DeltaArray:
        print("\n")

        formula = entry
        #------------------ Density and Volume --------------------
        print("------------------ Density and Volume --------------------")
        Density = GetDesity(f"Final_relaxedStructure_{formula}.cif")
        print (f"Density: {Density}")
        Volume = getVolume(f"Final_relaxedStructure_{formula}.cif")
        print(f"Volume: {Volume}")

        #------------------ Lattice Parameters --------------------
        print("------------------ Lattice Parameters --------------------")
        LatticeParamerters = getLatticeParameters(f"Final_relaxedStructure_{formula}.cif")
        print("\n")
        print(f"Lattice Parameters{LatticeParamerters}")

        #----------------------- Composition -----------------------
        print("----------------------- Composition -----------------------")
        compositionResults = getComposition(f"Final_relaxedStructure_{formula}.cif")
        print("\n")
        print(f"Composition Results: {compositionResults}")

        #----------------------- Space Group -----------------------
        print("----------------------- Space Group -----------------------")
        spaceGroup = getSpaceGroup(f"Final_relaxedStructure_{formula}.cif")
        print("\n")
        print(f"Space group: {spaceGroup}")

        #------------------- Local geometry -------------------
        # localGeometry = getLocalGeometry(f"Final_relaxedStructure_{formula}.cif")
        # print("\n")
        # print(f" Local geometry: {localGeometry}")

        #------------------ All Bond Lengths ------------------
        print("------------------ All Bond Lengths ------------------")
        BondLengths = GetbondLengths(f"Final_relaxedStructure_{formula}.cif")
        print("\n")
        print(f"Bond Lengths: {BondLengths}")

        #------------------ Prototype Matching ----------------
        print("------------------ Prototype Matching ----------------")
        prototypeGuess = prototypeMatching(f"Final_relaxedStructure_{formula}.cif")
        print("\n")
        print(f"Best Prototype Guess: {prototypeGuess}")


        print("\n")
        print("----------------------------------------------------- Slab Analisis -------------------------------------------")
        




