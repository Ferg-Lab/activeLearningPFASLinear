import os, sys, time
import numpy as np
import pandas as pd 
#import selfies
from tqdm import tqdm

#import selfies_024 as sf
import selfies as sf

#### ZINC training data files used for VAE
csc_df = pd.read_csv('../Data/VAE_data/csc-library-SMILES.csv')

#### Extract SELFIES of training samples
selfies_list = csc_df['SELFIES'].tolist()

#### Extract alphabets in SELFIES
alphabet_list = pd.read_csv('../Data/VAE_data/alphabet-list.csv')['alphabets'].tolist()

#### One hot encoding SELFIES max length.
selfies_len_list = csc_df['SELFIES'].apply(lambda selfi: selfi.count('[')).tolist()
selfies_len_max = max(selfies_len_list)

hot_dt = np.dtype(bool)

def len_selfies(selfies):
    
    return selfies.count("[") + selfies.count(".")

# filter dataframe using alphabets of the molecules
def check_molecules_alphabets_exists(df):
    
    """
    Check if alphabets of input SMILES in SELFIES format has the alphabets used in training.
    
    Parameters
    ----------
    
    df: DataFrame
        Dataframe containing SMILES of molecules being explored.
    
    Returns
    -------
    
    alpha_exists: List[Bool]
        Return in bool depending on whether the alphabets of SMILES exist.
    
    """

    smiles_list = df['smiles'].values
    
    alpha_exists = []
    for smiles in smiles_list:
    
        #print('--> Translating SMILES to SELFIES...')
        selfies_molecule = sf.encoder(smiles)

        #print(selfies_molecule)
    
        len_seflies = sf.len_selfies(selfies_molecule)
    
        #print(len_seflies)

        symbols_selfies = list(sf.split_selfies(selfies_molecule))
    
        #print(symbols_selfies)
    
        #print('Finished translating SMILES to SELFIES.')
            
        alpha_exists.append(all(elem in alphabet_list  for elem in symbols_selfies))
        
    

    return alpha_exists


def get_selfie_and_smiles_encodings_for_dataset(smilesDf):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    Parameters
    ----------
    
    input:
        smilesDf: Dataframe. 
            Column's name must be 'smiles'.
            
    Returns
    -------
    
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = smilesDf

    smiles_list = np.asanyarray(df.smiles)
    smiles_alphabet = list(set(''.join(smiles_list)))    
    smiles_alphabet.append(' ')  # for padding    
    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    largest_selfies_len = max(len_selfies(s) for s in selfies_list)
    
    print('Finished translating SMILES to SELFIES.')

    return selfies_list, alphabet_list, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len


def selfies_to_hot(molecule, largest_smile_len, alphabet):
    
    """
    Go from a single selfies string to a one-hot encoding.
    This is similar to that used in training the VAE.
    
    Parameters
    ----------
    
    molecule: String
        Input SELFIE string
        
    largest_smile_len:
        Length of the largest SMILE molecule
        
    alphabet: List
        Alphabets of SELFIES used in training.
    
    Returns
    -------
        One hot encoded vector.
    
    """
    
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input smile
    len_of_molecule=len(molecule)-len(molecule.replace('[',''))
    for _ in range(largest_smile_len-len_of_molecule):
        molecule+='[epsilon]'

    selfies_char_list_pre=molecule[1:-1].split('][')
    selfies_char_list=[]
    for selfies_element in selfies_char_list_pre:
        selfies_char_list.append('['+selfies_element+']')   

    integer_encoded = [char_to_int[char] for char in selfies_char_list]
    
    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
            
    return np.array(onehot_encoded, dtype=hot_dt)


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    
    """
    Convert a list of selfies strings to a one-hot encoding
    
    Parameters
    ----------
    selfies_list: List[Strings]
        Input list containing SELFIES of probes to be converted to one hot encoded vector.
        
    largest_molecule_len: Int
        Largest molecule in the search space.
        
    alphabet: List[Strings]
        Alphabets of SELFIES used in training.
        
    Returns
    -------
    Array of one hot encoded vectors for each SELFIE in the input list.
    
    """
    
    hot_list = []
    for selfiesI in tqdm(selfies_list):
        onehot_encoded = selfies_to_hot(selfiesI, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list, dtype=hot_dt)

def is_correct_smiles(smiles):
    
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    
    if smiles == "":
        return False

    try:
        return CChem.MolFromSmiles(smiles, sanitize=False) is not None
    except Exception:
        return False