import os
import shutil
import numpy as np
import pandas as pd
import json

def clean_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_parquet = os.path.join(base_dir, "df_lagu_mellow_indo.parquet")
    path_embed = os.path.join(base_dir, "embeddings_lagu_mellow.npy")
    path_faiss = os.path.join(base_dir, "index_lagu_mellow.faiss")

    print(f"Loading {path_parquet} and {path_embed}...")
    df = pd.read_parquet(path_parquet)
    embeddings = np.load(path_embed)
    print(f"Original df shape: {df.shape}")
    print(f"Original embeddings shape: {embeddings.shape}")

    # Create backups
    print("Creating backups...")
    shutil.copy2(path_parquet, path_parquet + ".bak")
    shutil.copy2(path_embed, path_embed + ".bak")
    if os.path.exists(path_faiss):
        shutil.copy2(path_faiss, path_faiss + ".bak")
    print("Backups created successfully.")

    # Load bogus indices from audit
    # We identified exactly 85 bogus indices:
    # 82 from the foreign unaccounted list (excluding 107 Screen and 162 REM)
    # plus idx 123 (Andrew & Noah Band), idx 319 (Alextbh - Nothing's Gonna Hurt You Baby), idx 410 (Gamelan Angklung - Segera Madu)
    from find_unaccounted import unaccounted
    bogus_indices = set()
    for u in unaccounted:
        if u['idx'] not in (107, 162):
            bogus_indices.add(u['idx'])
    
    bogus_indices.add(123)
    bogus_indices.add(319)
    bogus_indices.add(410)

    print(f"Total bogus indices identified for removal: {len(bogus_indices)}")

    # Create mask for clean tracks
    clean_mask = [i not in bogus_indices for i in range(len(df))]
    df_clean = df[clean_mask].reset_index(drop=True)
    embeddings_clean = embeddings[clean_mask]

    print(f"Cleaned df shape: {df_clean.shape}")
    print(f"Cleaned embeddings shape: {embeddings_clean.shape}")

    # Verify no foreign / classical / mismatch tracks remain
    foreign_keywords = [
        'orchestra', 'choir', 'corps', 'regiment', 'bart wolffe', 'music my pet',
        'chopin', 'haydn', 'beethoven', 'camille', 'saint-sa', 'music for little people',
        'benny goodman', 'billy connolly', 'mike birbiglia', 'andrew dice clay',
        'juice music', 'jet black crayon', 'peter kater', 'palace music', 'jenny wurlitzer',
        'erich wolfgang korngold', 'david s. ware', 'david roth', 'david parmley',
        'gingerbread patriots', 'joe locascio', 'wesley sprayue', 'summer lawns',
        'django reinhardt', 'hebert rasgado', 'ahmad jamal', 'ill ease', 'danny flanigan'
    ]

    for idx, r in df_clean.iterrows():
        combined = (str(r['name']) + ' ' + str(r['artists'])).lower()
        if any(k in combined for k in foreign_keywords):
            raise ValueError(f"Found unexpected foreign keyword in clean df at [{idx}]: {r['name']} | {r['artists']}")

    # Save cleaned parquet
    print(f"Saving cleaned DataFrame to {path_parquet}...")
    df_clean.to_parquet(path_parquet, index=False)

    # Save cleaned embeddings
    print(f"Saving cleaned embeddings to {path_embed}...")
    np.save(path_embed, embeddings_clean)

    # Rebuild FAISS index
    try:
        import faiss
        print("Rebuilding FAISS index...")
        dim = embeddings_clean.shape[1]
        index = faiss.IndexFlatIP(dim)
        # Normalize vectors for inner product (cosine similarity)
        norms = np.linalg.norm(embeddings_clean, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embed_norm = (embeddings_clean / norms).astype(np.float32)
        index.add(embed_norm)
        faiss.write_index(index, path_faiss)
        print(f"FAISS index rebuilt and saved with {index.ntotal} vectors.")
    except ImportError:
        print("FAISS not installed or available, skipped FAISS index file rewrite.")

    print("\nSUCCESS: Data sanitization and artifact synchronization complete!")
    return len(df_clean)

if __name__ == "__main__":
    clean_artifacts()
