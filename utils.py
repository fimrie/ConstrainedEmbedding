from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField

def get_frags(full_mol, clean_frag):
    """ aligns the 3D coordinates of two unconnected fragments to the
    coordinates of a reference molecule that contains the fragments.
    Arguments
      - full_mol: the reference molecule
      - clean_frag: the fragments without exit vectors
    """
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()

    if linker_len == 0:
        return full_mol

    Chem.Kekulize(full_mol, clearAromaticFlags=True)

    all_frags = []
    all_frags_lengths = []

    if len(matches)>0:
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            linker = Chem.Mol(mol_rw)
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                all_frags.append(frags)
                all_frags_lengths.append(len(Chem.rdmolops.GetMolFrags(frags)))
                if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                    return frags

    return all_frags[np.argmax(all_frags_lengths)]


def ConstrainedEmbed_Slack(mol, core, useTethers=True, tdist=0.25, coreConfId=-1, randomseed=2342,
                     getForceField=UFFGetMoleculeForceField, **kwargs):
    """ generates an embedding of a molecule where part of the molecule
    is constrained to have particular coordinates
    Arguments
      - mol: the molecule to embed
      - core: the molecule to use as a source of constraints
      - useTethers: (optional) if True, the final conformation will be
          optimized subject to a series of extra forces that pull the
          matching atoms to the positions of the core atoms. Otherwise
          simple distance constraints based on the core atoms will be
          used in the optimization.
      - tdist: (optional) if useTethers==True, a distance constraint 
          between the atoms and the positions of the core atoms during
          the optimization procedure. 
      - coreConfId: (optional) id of the core conformation to use
      - randomSeed: (optional) seed for the random number generator
    """
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    ci = AllChem.EmbedMolecule(mol, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    if ci < 0:
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]

    if not useTethers:
        # clean up the conformation
        ff = getForceField(mol, confId=0)
        for i, idxI in enumerate(match):
            for j in range(i + 1, len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.)
        ff.Initialize()
        n = 4
        more = ff.Minimize()
        while more and n:
            more = ff.Minimize()
            n -= 1
        # rotate the embedded conformation onto the core:
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
        ff = getForceField(mol, confId=0)
        conf = core.GetConformer()
        for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, match[i], 0, tdist, 100.)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    mol.SetProp('EmbedRMS', str(rms))
    return mol