from rdkit import Chem

def validate_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        return False
    return Chem.MolFromSmiles(smiles) is not None

def validate_batch(smiles_list):
    valid = [s for s in smiles_list if validate_smiles(s)]
    total = len(smiles_list)
    cnt = len(valid)
    pct = (100 * cnt / total) if total > 0 else 0.0
    print(f"Valid: {cnt}/{total} ({pct:.1f}%)")
    return valid

if __name__ == '__main__':
    test_smiles = ['CCO', 'C1=CC=CC=C1', 'invalid_smiles', 'CC(C)C']
    validate_batch(test_smiles)
