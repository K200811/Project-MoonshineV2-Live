from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

import numpy as np

import json

import time

start_time = time.perf_counter()

with open("moonshine_data.json", "r") as f:
    data = json.load(f)

data["payload"]["current_stage"] = "EpsilonProcess"

with open("moonshine_data.json", "w") as f:
    json.dump(data, f, indent=2)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
        


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
        data["payload"]["logs"]["warnings"].append("No prototype matches")
        return ("No Matches")
    matches = sorted(matches, key=lambda x: x["rms_score"])

    for rank, match in enumerate(matches, 1):
        print(f"Rank {rank}: {match['mp_id']} ({match['prototype_formula']}-type) | RMS Distortion: {match['rms_score']:.4f} Å")
    return matches[0]



#______________________________________________________________________________________________________________________________________

def createSlabs (file, MILLERINDEX,filename):
    structure = Structure.from_file(file)

    millerIndex = MILLERINDEX
    minSlabSize = 10.0
    minVacuumSize = 15.0

    slabGenerator = SlabGenerator(
        initial_structure=structure,
        miller_index=millerIndex,
        min_slab_size=minSlabSize,
        min_vacuum_size=minVacuumSize,
        center_slab=True   
    )

    allSlabs = slabGenerator.get_slabs()
    print(f"Generated {len(allSlabs)} slabs with termination for the miller Index of: {millerIndex} ")

    for i, slab in enumerate(allSlabs):
        slab.to(filename)

    return len(allSlabs) 

#____________________________________________________________________________________________


def slabAnalisis(file, MILLERINDEX):

    structure = Structure.from_file(file)

    slab = Slab(
        lattice=structure.lattice,
        species=structure.species,
        coords=structure.frac_coords,
        miller_index= MILLERINDEX,
        oriented_unit_cell= structure,
        shift=0,
        scale_factor=np.eye(3, dtype=int)
    )
    
    isSymmetric = slab.is_symmetric()
    isPolar = slab.is_polar()
    slabNormal = slab.normal
    slabShift = slab.shift
    slabScaleFactor = slab.scale_factor
    slabCenterOfMass = slab.center_of_mass

    c_coords = [round(site.frac_coords[2],2) for site in slab]
    layers = sorted(set(c_coords))

    c2_coords = np.array([site.frac_coords[2] for site in slab])
    topZ = c2_coords.max()
    botZ = c2_coords.min()
    tolerance = 0.05

    top_atoms = [slab[i].species_string for i in range(len(slab)) if abs(c2_coords[i] - topZ) < tolerance]
    bot_atoms = [slab[i].species_string for i in range(len(slab)) if abs(c2_coords[i] - botZ) < tolerance]

    surface_sites = slab.get_surface_sites()

    asf = AdsorbateSiteFinder(slab)
    adsorption_sites = asf.find_adsorption_sites()

    surfaceArea = slab.surface_area

    result = {
        "Is Symmetric": isSymmetric,
        "Is Polar": isPolar,
        "Slab Normal": slabNormal,
        "Slab Shift": slabShift,
        "Slab Scale Factor": slabScaleFactor,
        "Slab Center Of Mass": slabCenterOfMass,
        "Slab Layers": layers,
        "Top Termination Atoms": top_atoms,
        "Bottom Termination Atoms": bot_atoms,
        "Surface Sites": surface_sites,
        "Adsorption Sites": adsorption_sites,
        "Surface Area": surfaceArea
    }

    return result






        




if __name__ == '__main__':

    DeltaArray = []
    with open("DeltaFile", "r", encoding = "utf-8") as file:
        DeltaArray = json.load(file)
        UnitCellAnalisisArray = []
        slabAnalisisArray = []
        EArray = []

    for entry in DeltaArray:
        for j in range (5):
        

            print("\n")

            formula = entry
            #------------------ Density and Volume --------------------
            print("------------------ Density and Volume --------------------")
            Density = GetDesity(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print (f"Density: {Density}")
            Volume = getVolume(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print(f"Volume: {Volume}")

            #------------------ Lattice Parameters --------------------
            print("------------------ Lattice Parameters --------------------")
            LatticeParamerters = getLatticeParameters(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print("\n")
            print(f"Lattice Parameters{LatticeParamerters}")

            #----------------------- Composition -----------------------
            print("----------------------- Composition -----------------------")
            compositionResults = getComposition(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print("\n")
            print(f"Composition Results: {compositionResults}")

            #----------------------- Space Group -----------------------
            print("----------------------- Space Group -----------------------")
            spaceGroup = getSpaceGroup(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print("\n")
            print(f"Space group: {spaceGroup}")

            #------------------- Local geometry -------------------
            # localGeometry = getLocalGeometry(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            # print("\n")
            # print(f" Local geometry: {localGeometry}")

            #------------------ All Bond Lengths ------------------
            print("------------------ All Bond Lengths ------------------")
            BondLengths = GetbondLengths(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print("\n")
            print(f"Bond Lengths: {BondLengths}")

            #------------------ Prototype Matching ----------------
            print("------------------ Prototype Matching ----------------")
            prototypeGuess = prototypeMatching(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif")
            print("\n")
            print(f"Best Prototype Guess: {prototypeGuess}")

            UnitCellAnalisisArray.append({
                "Formula": formula,
                "Density": Density,
                "Volume": Volume,
                "Lattice Parameters": LatticeParamerters,
                "Composition": compositionResults,
                "Space Group": spaceGroup,
                "Bond Lenghts": BondLengths,
                "PrototypeMatchingGuess": prototypeGuess
            })

            print("\n")
            print("----------------------------------------------------- Slab Analisis -------------------------------------------")
            #---------------------------------- Slab Creation ---------------------
            print("---------------------------- Slab Creation ---------------------")
            print("Creating Slabs of Miller Index (1,1,1)")
            num_slabs = createSlabs(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif", (1,1,1), f"Slab_{j}_{entry['Formula']}_(1,1,1).cif")
            #------------------------------------------------------------
            print("Creating Slabs of Miller Index (1,0,0)")
            num_slabs2 = createSlabs(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif", (1,0,0), f"Slab_{j}_{entry['Formula']}_(1,0,0).cif")
            #------------------------------------------------------------
            print("Creating Slabs of Miller Index (1,1,0)")
            num_slabs3 = createSlabs(f"Final_relaxedStructure_{entry['Formula']}_{j}.cif", (1,1,0), f"Slab_{j}_{entry['Formula']}_(1,1,0).cif")

            total_number_of_slabs = num_slabs + num_slabs2 + num_slabs3
            print(f" Total Number Of Slabs {total_number_of_slabs}")

            #----------------------------- Symmetry Analisis ----------------------
            print("----------------------------- Slab Analisis ----------------------")

            

            for i in range (num_slabs):
                SlabAnalisisResult = slabAnalisis(f"Slab_{i}_{entry['Formula']}_(1,1,1).cif", (1,1,1))
                result = {
                    "File": f"Slab_{i}_{entry['Formula']}_(1,1,1).cif",
                    "Slab Analisis": SlabAnalisisResult
                }
                slabAnalisisArray.append(result)

            for i in range (num_slabs2):
                SlabAnalisisResult = slabAnalisis(f"Slab_{i}_{entry['Formula']}_(1,0,0).cif", (1,0,0))
                result = {
                    "File": f"Slab_{i}_{entry['Formula']}_(1,0,0).cif",
                    "Slab Analisis": SlabAnalisisResult
                }
                slabAnalisisArray.append(result)

            for i in range (num_slabs3):
                SlabAnalisisResult = slabAnalisis(f"Slab_{i}_{entry['Formula']}_(1,1,0).cif", (1,1,0))
                result = {
                    "File": f"Slab_{i}_{entry['Formula']}_(1,1,0).cif",
                    "Slab Analisis": SlabAnalisisResult
                }
                slabAnalisisArray.append(result)
                
            print (slabAnalisisArray)
            EArray.append(formula)




    with open("EpsilonFile_UnitCellAnalisis", "w") as json_file:
        json.dump(UnitCellAnalisisArray, json_file, indent=4, cls=NumpyEncoder)

    with open("EpsilonFile_SlabAnalisis", "w") as json_file:
        json.dump(slabAnalisisArray, json_file, indent=4, cls = NumpyEncoder)


    end_time = time.perf_counter()
    execution_time = end_time - start_time

    data["payload"]["stage_timing"][4]["seconds"] = execution_time

    for i in range (len(EArray)):
        data["payload"]["candidates_in_system"] = []
        data["payload"]["candidates_in_system"].append({
        "formula": EArray[i], "index": i, "id": f"cand_{i}", "status": "Passed Epsilon"
        })
        data["payload"]["logs"]["failed_all"].append(EArray[i])

    with open ("moonshine_data.json", "w") as f:
        json.dump(data, f, indent=2)
        


    





