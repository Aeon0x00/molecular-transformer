from rdkit import Chem

def validate_smiles(smiles):
    """Проверяет валидность SMILES через RDKit"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def validate_batch(smiles_list):
    """Валидация списка SMILES"""
    valid = [s for s in smiles_list if validate_smiles(s)]
    total = len(smiles_list)
    valid_count = len(valid)
    print(f"Valid: {valid_count}/{total} ({100*valid_count/total:.1f}%)")
    return valid

if __name__ == '__main__':
    test_smiles = ['CCO', 'C1=CC=CC=C1', 'invalid_smiles', 'CC(C)C']
    validate_batch(test_smiles)
