We will begin by trying to understand how to make linear classifiers ([[Linear Classification]]) more powerful (better able to fit training examples) by adding additional features and thinking about practically useful ways of encoding information into feature vectors in a real context.

- It is important to make sure that the process does not create linearly dependent features.
- **Caution!**
	- The Goal is generalization
	- if transform $\phi(\vec{x})$ is too flexible, we will fit $S_{n}$, but the error on $S_{n'}$, will be large
		- Adding more data to $S_{n}$ should help choosing the right model.

- **Stability**
	- The Model only changes a little with the new data if it is already trained on a representative set

# Representations

## 3D Geometry
- Cartesian coordinates
- Internal coordinates

## Graph
- Lewis Structure
- Collection of vertices/nodes and edges/links
	- $\Gamma =\{\gamma_{1},\gamma_{2}, \dots  \}$
	- $E \subset \{ e_{ij} = (\gamma_{i},\gamma_{j}) :\gamma_{i},\gamma_{j} \in \Gamma \}$
- directed vs. undirected graph 
- Adjacency Matrix
	- encodes presence/absence of edges 


## String Representation 

- Sequence Based Representations
- SMILES 
- SMARTS
	- patterns
	- functional group
	- reactions
	- substitutions
- SELFIES
	- more robust than SMILES
	- always valid if following grammar rules
	- **Claim:** non-validity of SMILES can teach the model something
- InChl(Key):
	- non-human readable 
	- keys are simplified 27 characters long string
	- compact, standardized without loss of detail
- Fingerprints
	- structural
		- encode presence of a substructure (bit-vector)
	- circular
		- considers arrangement within a specified radius (over a certain amount of bonds)
		- **Morgan Fingerprint**
	- topological fingerprints
		- rings, paths, overall shape 
	- pharmacophore
		- spatial arrangement of pharmacological features
			- $H$-Bond donors/acceptors etc. 
- Discriptors
	- Topological discriptors
		- based on the graph 
		- representing the connectivity and arrangement of atoms within a molecule
		- does not consider the 3D geometry
	- Geometrical discriptors
		- Take into account the three dimensional arrangement of atoms in a molecule
	- Electronic descriptors 
		- capture information about the electronic structure
			- HOMO Energy, Density, Electron-negativity
	- Thermodynamic discriptors
		- Heat of formation, entropy, ... 
	- 



