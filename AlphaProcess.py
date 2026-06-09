import logging
import json
import sys

#____________________________________

# Dictionaries

Elements_Dictonary = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
  "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
  "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
  "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
  "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
  "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og" 
]






#______________________________________


#######################################################


#__________________________________

# Set up Starting Crystal Info

Referance_Crystal = "NH4H2PO4"

Referance_Crystal_Elements = ["N","H","H","P","O"]
Referance_Crystal_Subscripts = [1,4,2,1,4]

#____________________________________

#############################################################

#__________________________________

# Set up Logger

logging.basicConfig(
    filename='Logs.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)

#___________________________________

###################################################

#_______________________________________________________

# Checking that Crystal info was put in correctly

def SuperAlphaCheck():
    if len(Referance_Crystal_Elements) == len(Referance_Crystal_Subscripts):
        logging.info("Crystal Info Properly Inputed")
        print("Crystal Info Properly Inputed")
    else:
        logging.error("Crystal Info Not Properly Inputed")
        print("Crystal Info Not Properly Inputed")
        sys.exit()



#______________________________________________________________

#Create Alpha array and Refrance Array


Refrance_Array = []
Alpha_Array = []



def SuperAlphaFunction():

    for i in range(len(Referance_Crystal_Subscripts)):
        for j in range(Referance_Crystal_Subscripts[i]):
            Refrance_Array.append(Referance_Crystal_Elements[i])

    logging.info(f"Refrance Array: {Refrance_Array}")
    print(f"Refrance Array: {Refrance_Array}")


def AlphaFunction():

    for i in range (len(Refrance_Array)):
        temp = Refrance_Array.copy()
        print(temp)
        for j in range (len(Elements_Dictonary)):
            temp[i] = Elements_Dictonary[j]
            Alpha_Array.append(temp)
            temp = Refrance_Array.copy()

    logging.info(f"Alpha Array: {Alpha_Array}")
    logging.info(f"Alpha Array Length: {len(Alpha_Array)}")
    logging.info((f" Elements Length: {len(Elements_Dictonary)}"))
    logging.info(f" Refrance Array Length: {len(Refrance_Array)}")

    print(f"Alpha Array: {Alpha_Array}")
    print(f"Alpha Array Length: {len(Alpha_Array)}")
    print((f" Elements Length: {len(Elements_Dictonary)}"))
    print(f" Refrance Array Length: {len(Refrance_Array)}")




#___________________________________________________________

##############################################################



#_______________________________________________________________

# Main Function

if __name__ == "__main__":
    print("Results of Super Alpha Check: ")
    SuperAlphaCheck()
    SuperAlphaFunction()
    AlphaFunction()
#_____________________________________________________________

#JSON file creation

with open("Alpha_Arrays.json", "w") as json_file:
    json.dump(Alpha_Array, json_file, indent = 4)

logging.info("Alpha_Arrays.json created")

#______________________________________________________________


##################################################################



