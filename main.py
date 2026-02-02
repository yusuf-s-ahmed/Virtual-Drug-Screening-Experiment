import pandas as pd

xls = pd.ExcelFile('587184_file06.xlsx')

df = pd.read_excel(xls, 'Main training set 3')

from chembl_webresource_client.new_client import new_client

molecule = new_client.molecule

ids = df["chembl_id"].tolist()   # ← keep full list, no dropping rows
unique_ids = list(set(ids))      # ← internal lookup only

records = molecule.filter(
    molecule_chembl_id__in=unique_ids
).only(
    "molecule_chembl_id",
    "molecule_structures"
)

id_to_smiles = {
    r["molecule_chembl_id"]:
        (r.get("molecule_structures") or {}).get("canonical_smiles")
    for r in records
}
df["smiles"] = df["chembl_id"].map(id_to_smiles)

missing = df["smiles"].isna().sum()
print("Missing SMILES:", missing)

df[df["smiles"].isna()].head()

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# define function that transforms SMILES strings into ECFP6 fingerprints
def ECFP6_from_smiles(smiles,
                     R = 3,
                     L = 2**10,
                     use_features = False,
                     use_chirality = False):
    
    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius = R,
        nBits = L,
        useFeatures = use_features,
        useChirality = use_chirality
    )

    return np.array(feature_list).tolist()


from rdkit.Chem import RDKFingerprint

def extended_fp(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return RDKFingerprint(molecule)

def merge_fingerprints(fp1, fp2):
    """
    Inputs:
    
    - fp1 ... first fingerprint as a list
    - fp2 ... second fingerprint as a list
    
    Outputs:
    - merged_fp ... merged fingerprint as a list
    """
    merged_fp = np.concatenate((np.array(fp1), np.array(fp2)))
    return merged_fp.tolist()

X_train = np.array([
    ECFP6_from_smiles(smiles) for smiles in df['smiles']
])

y_train = df["label"].map({
    "posi": 1,
    "nega": 0
}).values

print(np.unique(y_train))
print(np.bincount(y_train))

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    "kernel": ["rbf"],
    "gamma": ["scale"]
}

svm = GridSearchCV(
    SVC(),
    param_grid,
    scoring = 'accuracy',
    cv = 10
)

svm.fit(X_train, y_train)

print(svm.best_params_)
print(svm.best_score_)

df_test = pd.read_excel(xls, "Main test set")
df_test.head()

missing_ids = df_test.loc[
    ~df_test["chembl_id"].isin(id_to_smiles),
    "chembl_id"
].unique().tolist()

if missing_ids:
    records = molecule.filter(
        molecule_chembl_id__in=missing_ids
    ).only(
        "molecule_chembl_id",
        "molecule_structures"
    )

    for r in records:
        mid = r["molecule_chembl_id"]
        smi = (r.get("molecule_structures") or {}).get("canonical_smiles")
        id_to_smiles[mid] = smi
        
df_test["smiles"] = df_test["chembl_id"].map(id_to_smiles)

print("Missing SMILES in test:", df_test["smiles"].isna().sum())
print(df_test["label"].value_counts())

X_test = np.array([
    ECFP6_from_smiles(s) for s in df_test["smiles"]
])

y_test = df_test["label"].map({
    "posi": 1,
    "nega": 0
}).values

from sklearn.metrics import accuracy_score, classification_report

y_pred = svm.best_estimator_.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
