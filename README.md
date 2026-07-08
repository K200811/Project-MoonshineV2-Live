Moonshine Algorithm
An automated end-to-end pipeline for AI-driven inorganic crystal structure discovery.

Project Moonshine is an algorithm designed to generate novel materials candidates based on variations of a starting crystal. It uses LLM guided filtering, rule based chemical screening, and neural network driven structural relaxation to accelerate the discovery of novel inorganic crystals structures, by reducing the time it takes to process candidates. 
Overview
Moonshine starts with a seed crystal that it expands out into variations, made by substituting each elemental index with every element from the periodic table. These are called Alpha Arrays and are carried through the system and eliminated as soon as they violate any filters.
Pipeline stages:
Alpha Processing Layer -- Uses basic arithmetic and array manipulation to generate a list of candidate arrays, based on the initial starting crystal given. 
Beta Processing Layer -- Uses two sets of the Gemma 3 LLM to generate and validate empirical formulas created based on the Alpha Arrays from the last layer. Any formula scored above 50 moves on to the next stage
Gamma Processing Layer -- Uses another LLM this time deepseek to generate the parameters used for initial crystal structure generation for each candidate using PyXtal. 500 structures are created and the top five with the lowest total Ev / atom move on to the next phase. Each candidate is then subjected to an initial relaxation using CHGNet to get it into a rough shape of its final structure. If the forces and stress values acting on the structure and lactic are less than 0.1 for each, the candidate structure moves onto the next layer. Next a more strict and refined CHGNet relaxation is applied that relaxes the rough structure into its final tight structure. All structures with forces and stress values less than 0.1 move on to the next layer. 
Delta Processing Layer -- The delta processing layer is used to validate the structures created by subjecting them to a suite of tests. The Gamma layer outputs 5 structures per candidate and Delta tests each and every one. The tests run are the following:  Atomic overlap, composition, coordination number plusability, oxidization number plusability, and structure matching against the materials project. Any structures that pass all tests move onto the final layer
Epsilon Processing Layer -- The epsilon processing layer is used to analyze the bulk unit cell structure created, and the slab structure variants created off of it in this layer. Unit cell analysis returns density, unit cell volume, lattice parameters, final space group symmetry, local atomic geometry, bond lengths, bond angles, and prototype classification. Then comes the slab analisis. First the slabs are created by pymatgen, the main library used in this entire project, and analyzed for the following: surface symmetry, polarity, surface normal vectors, scale factors, center of mass alignment, termination layer composition, candidate adsorption sites, and total surface area. Slabs are created along the three Miller indices; there are just numbers that tell how the crystal is sliced in 3d space in reference to the three axes, (1,0,0) (1,1,0) and (1,1,1).

Motivation
This project exists to quickly create candidates that could be novel new materials. Currently extensive amounts of compute are used to run advanced algorithms in search of autonomously finding new materials. This pipeline is meant to generate candidates with already a high likelihood of stability and validity for those computationally expensive algorithms to further evaluate. The secondary intent of this project is to create a way for the average person to delve in materials science discovery. Science and technology are fields that are innately collaborative and thrive off of many people working on the same problem. By providing a computationally less expensive alternative that anyone with a laptop can use, anyone who wants to test their materials ideas, or come up with something new now has a way to get started. 


Key Features
Usage of standard off the self LLM’s for advanced processes such as formula creation, validation, and crystal structure parameter generation. 
CHGNet structural relaxation that quickly creates high accuracy 3D structures
Architecture designed to filter out improbable candidates as quickly as possible
In depth structural analysis report







Installation
git clone https://github.com/K200811/Project-MoonshineV2-Live

Requirements:
Python version: 3.13.1
Dependencies: pyxtal, cygnet, pymatgen, ase, json, logging
external services: Need ollama installed and the models you want pulled. Change the local host address to what your ollama instance is running on. Also must get a API key from the Materials Project

Usage
Quick start
Go into the Alpha Processing Layer and fill in the information about your seed crystal as asked in lines 47 - 50. In the Reference_Crystal_Elements array just list your starting seed crystals elements in order, same for the subscripts array just instead of the elements, the subscript associated with each element, in the same order as before. Run each layer. At the end the structures labeled Finnal_Crystal_Structure along with a formula in the name are the structures that have been produced. The slab structures are labeled slabs. All structures are in CIF format and can be viewed in a viewer like VESTA.

Turn on moonshine server to see the UI. Program is not ment to be used with the UI

License
license type: MIT

