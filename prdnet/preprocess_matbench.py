import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocess import Pool
from ase.neighborlist import NewPrimitiveNeighborList
from torch_geometric.data import Data
from matminer.datasets import load_dataset
from sklearn.model_selection import KFold
from pymatgen.io.ase import AseAtomsAdaptor
import argparse

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    if (abs(sum(a_new * b_new) - 1.0) < 1e-5) or (abs(sum(a_new * b_new) + 1.0) < 1e-5):
        return True
    else:
        return False

def same_plane(a, b, c):
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        return True
    else:
        return False
       
def get_neighbors_and_vectors(atoms, cutoff=4.0, max_neighbors=12, add_lattice_to_edge=False):
    positions = atoms.positions
    cell = atoms.get_cell().array
    pbc = atoms.pbc
    nl = NewPrimitiveNeighborList(cutoffs=cutoff, self_interaction=False, bothways=True)
    nl.update(pbc, cell, positions)

    lattice_lengths = atoms.get_cell().lengths()
    max_lattice_lengths = max(lattice_lengths) + 1e-2
    nl1 = NewPrimitiveNeighborList(cutoffs=max_lattice_lengths, self_interaction=False, bothways=True)
    nl1.update(pbc, cell, positions)
    indices_of_atom0, offsets_of_atom0 = nl1.get_neighbors(0)
    r0 = positions[0]
    neigh = []
    for j, offset in zip(indices_of_atom0, offsets_of_atom0):
        if j == 0:
            rj = positions[j] + offset @ cell
            rel_vec = rj - r0
            dist = float(np.linalg.norm(rel_vec))
            neigh.append((dist, rel_vec))
    neigh.sort(key=lambda x: x[0])
    lat1_vec = neigh[0][1]
    start = 1
    for i in range(start, len(neigh)):
        lat2_vec = neigh[i][1]
        if not same_line(lat1_vec, lat2_vec):
            start = i
            break
    for i in range(start, len(neigh)):
        lat3_vec = neigh[i][1]
        if not same_plane(lat1_vec, lat2_vec, lat3_vec):
            break

    lat1_lat2_angle = np.dot(lat1_vec, lat2_vec)
    lat1_lat3_angle = np.dot(lat1_vec, lat3_vec)
    if lat1_lat2_angle < 0.0:
        lat2_vec = -lat2_vec
    if lat1_lat3_angle < 0.0:
        lat3_vec = -lat3_vec

    if np.dot(np.cross(lat1_vec, lat2_vec), lat3_vec) < 0.0:
        lat1_vec = -lat1_vec
        lat2_vec = -lat2_vec
        lat3_vec = -lat3_vec
    edge_index = []
    edge_dist = []
    edge_vec = []
    per_atom_rel_vectors = [[] for _ in range(len(atoms))]
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
       
        if len(indices) < max_neighbors:
            cell_len = atoms.get_cell().lengths()
            new_cut = max(cell_len) if cutoff < max(cell_len) else 2 * cutoff
            return get_neighbors_and_vectors(atoms, new_cut, max_neighbors)
        ri = positions[i]
        neigh = []
        for j, offset in zip(indices, offsets):
            rj = positions[j] + offset @ cell
            rel_vec = rj - ri
            dist = float(np.linalg.norm(rel_vec))
            neigh.append((int(j), dist, rel_vec))
        neigh.sort(key=lambda x: x[1])
        max_dist = neigh[max_neighbors - 1][1]
       
        for j, d, vec in neigh:
            if d <= max_dist:
                edge_index.append((j, i))
                edge_dist.append(d)
                edge_vec.append(vec)
                per_atom_rel_vectors[i].append(vec)
        if add_lattice_to_edge:
            edge_index.append((i, i))
            edge_dist.append(np.linalg.norm(lat1_vec))
            edge_vec.append(lat1_vec)
            per_atom_rel_vectors[i].append(lat1_vec)
            edge_index.append((i, i))
            edge_dist.append(np.linalg.norm(lat2_vec))
            edge_vec.append(lat2_vec)
            per_atom_rel_vectors[i].append(lat2_vec)
            edge_index.append((i, i))
            edge_dist.append(np.linalg.norm(lat3_vec))
            edge_vec.append(lat3_vec)
            per_atom_rel_vectors[i].append(lat3_vec)
   
               
    return edge_index, edge_dist, edge_vec, per_atom_rel_vectors, lat1_vec, lat2_vec, lat3_vec


def process_one_structure(args):
    idx, row, label_col, max_neighbors, add_lattice_to_edge = args
    try:
        pmg_struct = row["structure"]
        atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
        edge_index, edge_dist, edge_vec, rel_vectors, lat1_vec, lat2_vec, lat3_vec = get_neighbors_and_vectors(atoms, max_neighbors=max_neighbors, add_lattice_to_edge=add_lattice_to_edge)
        
        positions = atoms.positions  
        cell = atoms.get_cell().array 

        inv_cell = np.linalg.inv(cell)
        frac_coords = positions @ inv_cell
        frac_coords = frac_coords % 1.0

        y_val = row[label_col]
        y = torch.tensor(float(y_val), dtype=torch.float) if not pd.isna(y_val) else None
        partial_data = Data(
            x=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).view(-1, 1),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edgewise_euclidean_distance=torch.tensor(edge_dist, dtype=torch.float),
            y=y,
            lattice_vectors=torch.from_numpy(np.array([lat1_vec, lat2_vec, lat3_vec], dtype=np.float32)),
            edgewise_euclidean_distance_vec=torch.from_numpy(np.array(edge_vec, dtype=np.float32)),
            idx_in_df=int(idx),
            pos=torch.from_numpy(frac_coords).float(), 
            real_lattice_vectors=torch.from_numpy(cell).float(),
        )
        return idx, partial_data
    except Exception as e:
        print(f"[ERROR] Index {idx}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process matminer dataset with KFold and save to .data file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load from matminer")
    parser.add_argument("--max_neighbors", type=int, default=12, help="Maximum number of neighbors per atom (default: 12)")
    parser.add_argument("--out_dir", type=str, default="../data/matbench/processed_datasets", help="Output directory for processed .data files (default: ../data/matbench/processed_datasets)")
    parser.add_argument("--add_lattice_to_edge", action="store_true", help="Add lattice vectors as edges (default: False)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_dataset(args.dataset_name)
    dataset_to_label = {
        "matbench_expt_gap": "gap expt",
        "matbench_expt_is_metal": "is_metal",
        "matbench_glass": "gfa",
        "matbench_steels": "yield strength"
    }
    if args.dataset_name in dataset_to_label:
        label_col = dataset_to_label[args.dataset_name]
        if label_col not in df.columns:
            raise ValueError(f"Expected label column '{label_col}' not found in dataset '{args.dataset_name}'. Available columns: {list(df.columns)}")
    else:
        non_struct_cols = [c for c in df.columns if c != "structure"]
        if len(non_struct_cols) != 1:
            raise ValueError(f"Expected exactly one label column besides 'structure'. Found: {non_struct_cols}")
        label_col = non_struct_cols[0]
    print(f"Dataset: {args.dataset_name} | Label column: '{label_col}' | Samples: {len(df)}")
    # Perform KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=18012019)
    test_indices_per_fold = []
    for fold, (_, test_idx) in enumerate(kf.split(df)):
        test_indices_per_fold.append(set(test_idx))
        print(f"\nFold {fold} - Test set indices:")
        print(test_idx)
    # Add fold info to DataFrame
    for fold in range(5):
        df[f"fold_{fold}_is_test"] = df.index.isin(test_indices_per_fold[fold])
    # Prepare tasks: (idx, row, label_col, max_neighbors)
    tasks = [(idx, row, label_col, args.max_neighbors, args.add_lattice_to_edge) for idx, row in df.iterrows()]
    # Process structures in parallel to get partial data
    all_partial_data = [None] * len(df)
    with Pool(processes=56) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one_structure, tasks),
            total=len(tasks),
            desc="Processing structures"
        ))
    for res in results:
        if res is not None:
            idx, partial_data = res
            all_partial_data[idx] = partial_data
    # Add fold flags
    all_data = []
    for idx, data in enumerate(all_partial_data):
        if data is not None:
            for fold in range(5):
                setattr(data, f"fold_{fold}_is_test", bool(df.loc[idx, f"fold_{fold}_is_test"]))
            all_data.append(data)
    print(f"\n✅ Successfully processed {len(all_data)} / {len(df)} samples.")
    # Save to single file
    output_path = os.path.join(args.out_dir, f"{args.dataset_name}.data")
    torch.save(all_data, output_path)
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    main()
