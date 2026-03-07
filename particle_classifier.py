# Train classifier to evaluate sampling quality over multiple random seeds
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store_true', help='Making data for classifier training and evaluation')
parser.add_argument('--run', action='store_true', help='Run the classifier training and evaluation')
args = parser.parse_args()

# Utility functions
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def train_classifier(X_train, y_train, X_test, y_test, seed):
    """Train Random Forest classifier and return AUC score"""
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=50, 
        max_depth=5, 
        random_state=seed, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc, y_pred_proba, clf

def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Map 3 continuous values u to a beta vector within the unit sphere.
    """
    u = np.asarray(u, dtype=np.float64)
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    beta = np.tanh(norm + eps) * (u / (norm + eps))
    return beta

def load_events(path: str):
    """
    Load events from a .npy file, handling different possible formats.
    """
    arr = np.load(path, allow_pickle=True)
    
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]
    
    raise ValueError(f"Unrecognized format in {path}")

def sanitize_event(ev, me=0.000511):
    """
    Standardize event data into a consistent format.
    
    Return: (pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz)
    """
    ev = np.asarray(ev)

    # Case A: generated / explicit PDG format: [pdg, E, betax, betay, betaz, x, y, z]
    if ev.ndim == 2 and ev.shape[1] >= 8:
        pdg = ev[:, 0].astype(np.int64, copy=False)
        Eabs = np.abs(ev[:, 1].astype(np.float64, copy=False))
        betax = ev[:, 2].astype(np.float64, copy=False)
        betay = ev[:, 3].astype(np.float64, copy=False)
        betaz = ev[:, 4].astype(np.float64, copy=False)
        x = ev[:, 5].astype(np.float64, copy=False)
        y = ev[:, 6].astype(np.float64, copy=False)
        z = ev[:, 7].astype(np.float64, copy=False)

        beta = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)
        pvec = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
        E_signed = np.where(pdg == -11, -Eabs, Eabs)

        return (pdg, px, py, pz, Eabs, E_signed, beta_mag,
                x, y, z, betax, betay, betaz)

    # Case B: real guineapig format [E_signed, betax, betay, betaz, x, y, z]
    if ev.ndim == 2 and ev.shape[1] >= 7:
        E_signed = ev[:, 0].astype(np.float64, copy=False)
        betax = ev[:, 1].astype(np.float64, copy=False)
        betay = ev[:, 2].astype(np.float64, copy=False)
        betaz = ev[:, 3].astype(np.float64, copy=False)
        x = ev[:, 4].astype(np.float64, copy=False)
        y = ev[:, 5].astype(np.float64, copy=False)
        z = ev[:, 6].astype(np.float64, copy=False)

        pdg = np.where(E_signed >= 0.0, 11, -11).astype(np.int64)
        Eabs = np.abs(E_signed)
        beta = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)
        pvec = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]

        return (pdg, px, py, pz, Eabs, E_signed, beta_mag,
                x, y, z, betax, betay, betaz)

    # fallback: empty
    empty = np.array([], dtype=np.float64)
    return (
        empty.astype(np.int64),
        empty, empty, empty,
        empty, empty, empty,
        empty, empty, empty,
        empty, empty, empty
    )

def extract_species(events, pdgs=None, me=0.000511):
    """
    Extract physical quantities for specified particle species from a list of events.
    """
    mult = np.zeros(len(events), dtype=np.int64)
    px_list, py_list, pz_list = [], [], []
    E_list, Esigned_list, bmag_list = [], [], []
    x_list, y_list, z_list = [], [], []
    bx_list, by_list, bz_list = [], [], []

    for i, ev in enumerate(events):
        pdg, px, py, pz, Eabs, E_signed, bmag, x, y, z, betax, betay, betaz = sanitize_event(ev, me=me)

        if pdgs is None:
            sel = np.ones(len(px), dtype=bool)
        else:
            sel = np.zeros(len(px), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)

        mult[i] = int(np.sum(sel))

        if np.any(sel):
            px_list.append(px[sel]); py_list.append(py[sel]); pz_list.append(pz[sel])
            E_list.append(Eabs[sel])
            Esigned_list.append(E_signed[sel])
            bmag_list.append(bmag[sel])
            bx_list.append(betax[sel]); by_list.append(betay[sel]); bz_list.append(betaz[sel])

            if x.size:
                x_list.append(x[sel]); y_list.append(y[sel]); z_list.append(z[sel])

    def cat_or_empty(lst):
        return np.concatenate(lst) if len(lst) else np.array([], dtype=np.float64)

    px_all = cat_or_empty(px_list)
    py_all = cat_or_empty(py_list)
    pz_all = cat_or_empty(pz_list)
    p_all  = np.sqrt(px_all**2 + py_all**2 + pz_all**2)
    pt_all = np.sqrt(px_all**2 + py_all**2)

    E_signed_all = cat_or_empty(Esigned_list)

    return {
        "mult": mult,
        "px": px_all, "py": py_all, "pz": pz_all, "p": p_all, "pt": pt_all,
        "E": cat_or_empty(E_list),
        "E_abs": cat_or_empty(E_list),
        "E_signed": E_signed_all,
        "beta_mag": cat_or_empty(bmag_list),
        "x": cat_or_empty(x_list), "y": cat_or_empty(y_list), "z": cat_or_empty(z_list),
        "betax": cat_or_empty(bx_list),
        "betay": cat_or_empty(by_list),
        "betaz": cat_or_empty(bz_list),
    }

def prepare_classifier_data(real_sp, gen_sp, output_path, max_samples=100000):
    """
    Prepare data for classifier training to evaluate sampling quality.
    
    7 features: E, px, py, pz, betax, betay, betaz
    
    Args:
        real_sp: Dictionary with real data
        gen_sp: Dictionary with generated data
        output_path: Path to save the data file
        max_samples: Maximum samples per class (None = use all)
    
    Returns:
        Dictionary with X (features), y (labels), and feature names
    """
    # Extract 7 features for real data
    real_features = np.column_stack([
        real_sp['E'],
        real_sp['px'],
        real_sp['py'],
        real_sp['pz'],
        real_sp['betax'],
        real_sp['betay'],
        real_sp['betaz']
    ])
    
    # Extract 7 features for generated data
    gen_features = np.column_stack([
        gen_sp['E'],
        gen_sp['px'],
        gen_sp['py'],
        gen_sp['pz'],
        gen_sp['betax'],
        gen_sp['betay'],
        gen_sp['betaz']
    ])
    
    # Remove any rows with NaN or Inf
    real_mask = np.all(np.isfinite(real_features), axis=1)
    gen_mask = np.all(np.isfinite(gen_features), axis=1)
    
    real_features = real_features[real_mask]
    gen_features = gen_features[gen_mask]
    
    print(f"Real data: {len(real_features)} samples")
    print(f"Generated data: {len(gen_features)} samples")
    
    # Downsample if requested
    if max_samples is not None:
        if len(real_features) > max_samples:
            indices = np.random.choice(len(real_features), max_samples, replace=False)
            real_features = real_features[indices]
            print(f"  Downsampled real to {max_samples}")
        
        if len(gen_features) > max_samples:
            indices = np.random.choice(len(gen_features), max_samples, replace=False)
            gen_features = gen_features[indices]
            print(f"  Downsampled generated to {max_samples}")
    
    # Create labels: 0 = real, 1 = generated
    real_labels = np.zeros(len(real_features), dtype=np.int32)
    gen_labels = np.ones(len(gen_features), dtype=np.int32)
    
    # Combine data
    X = np.vstack([real_features, gen_features])
    y = np.concatenate([real_labels, gen_labels])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Feature names
    feature_names = ['E', 'px', 'py', 'pz', 'betax', 'betay', 'betaz']
    
    # Save data
    data_dict = {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'n_real': len(real_features),
        'n_generated': len(gen_features),
        'description': 'Classifier data: 0=real, 1=generated. Lower AUC = better sampling quality.'
    }
    
    np.save(output_path, data_dict)
    print(f"\n✓ Saved classifier data to: {output_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {feature_names}")
    print(f"  Real (0): {np.sum(y==0)}, Generated (1): {np.sum(y==1)}")
    print(f"\nNote: Lower classifier AUC indicates better sampling quality")
    
    return data_dict

# Data preparation
if args.data:
    REAL_DATA_PATH = "/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy"
    GEN_DATA_PATH = f"/work/submit/anton100/msci-project/FCC-BB-GenAI/new_10/generated_events.npy"

    print(f"Real data path: {REAL_DATA_PATH}")
    print(f"Generated data path: {GEN_DATA_PATH}")

    print("Loading real data...")
    real_events = load_events(REAL_DATA_PATH)
    n_real = len(real_events)
    print(f"Loaded {n_real} real events")

    print("\nLoading generated data...")
    gen_events = load_events(GEN_DATA_PATH)
    n_gen = len(gen_events)
    print(f"Loaded {n_gen} generated events")

    print(f"\nStat: {n_real} real events, {n_gen} generated events")

    species_list = [
    {"name": "e−",  "pdgs": [11],   "tag": "eminus"},
    {"name": "e+",  "pdgs": [-11],  "tag": "eplus"},
    {"name": "all", "pdgs": None,   "tag": "all"},
    ]

    real_data = {}
    gen_data = {}

    for sp in species_list:
        print(f"\nExtracting {sp['name']}...")
        
        real_data[sp['tag']] = extract_species(real_events, sp['pdgs'])
        gen_data[sp['tag']] = extract_species(gen_events, sp['pdgs'])
        
        print(f"  Real: {len(real_data[sp['tag']]['E'])} particles")
        print(f"  Gen:  {len(gen_data[sp['tag']]['E'])} particles")

    print("\n✓ All species extracted!")

    # Save the data
    print("\nPreparing classifier data...")
    tag = 'all'
    real_sp = real_data[tag]
    gen_sp = gen_data[tag]

    output_path = f'/work/submit/anton100/msci-project/FCC-BB-GenAI/new_10/classifier_data.npy'

    classifier_data = prepare_classifier_data(
        real_sp=real_sp,
        gen_sp=gen_sp,
        output_path=output_path
    )

    print(f"\n✓ Data preparation complete!, saved to: {output_path}")


# Train and evaluate classifier
if args.run:
    path = f'/work/submit/anton100/msci-project/FCC-BB-GenAI/new_10/classifier_data.npy'
    data = np.load(path, allow_pickle=True).item()
    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # Train over 10 different random seeds
    n_runs = 10
    auc_scores = []

    for run in range(n_runs):
        seed = random.randint(0, 100000)
        set_seed(seed)
        
        print(f"Run {run+1}/{n_runs} | Seed: {seed}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        auc, y_pred_proba, clf = train_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, seed
        )
        
        auc_scores.append(auc)
        print(f"  AUC: {auc:.4f}\n")

    # Compute statistics
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print(f"FINAL RESULTS OVER {n_runs} RUNS")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Min AUC:  {np.min(auc_scores):.4f}")
    print(f"Max AUC:  {np.max(auc_scores):.4f}")
    print(f"\nAll AUC scores: {[f'{auc:.4f}' for auc in auc_scores]}")

    # Feature importance from the last trained model
    importances = clf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Last Run)")
    print(f"{'='*60}")
    for feat, imp in feature_importance:
        print(f"  {feat:10s}: {imp:.4f}")