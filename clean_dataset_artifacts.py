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

    # 85 Spotify IDs of anomalous / mismatched foreign tracks identified from data audit
    BOGUS_TRACK_IDS = {
        '03jKDBjArXob1kMHVESUtI', '08pOc6inhy9XslqTNmsz9z', '0F0tsMNFoL45f030iSVRUj', '0FFeBP9k81itfG5JhHOIRM',
        '0GmP86RrkXpQS2Ex6pYevb', '0QKC44QYdJaQThknEAf2lY', '0Qs8vlwMO7M22jOpwWWVEs', '0SnwgJCaudo0yhj7HIZtOl',
        '0Tb87SaMgquop8HC1VJvad', '0UwdlK28gKCVaDOLL4tmV6', '0XSRBlP49MgzBCYtGK2NYE', '0bLcodI4bqgbbLGuej8Ibv',
        '0hoshYwNeUtOfGBgL3Cert', '0z1fdv7C3PUr1jpL7T6X0S', '1LJNYc4wV82VIJ856D46MX', '1NjbNArND1ZztO9VrzGZJB',
        '1PsCgg7l0DLaJnCCVQuMiB', '1Tt9pj1BUtKCY0c4xyh6fi', '1X6X93i1SA6EUPBOHLZ0F8', '1buXb5ywVselhjicvyGc9u',
        '1eS5Uv1Hx4cKxgWxejTNVp', '1ez3bXkni875o0io9zblUQ', '1zGzgtKLLwtxdo3h85cv4W', '25N9UhQtkdyQovEm7UAXBs',
        '2EsZbef25s6YYfrbTAClLi', '2Fg0dMa5fbDVA5oCBsPDvk', '2FhUS6Y6P9mZ47QV42E6C6', '2LPibirlWTAdLuw0UW72au',
        '2NKTvzKhxguEOWle0fhETL', '2W8PhL1Jn1msvdVkFzW0V2', '2fQAmwe5cCu0NV2JsK9PjP', '2p0RzJ7LSgLx68JBVdxeuB',
        '34k2BkM0VECjiLuho1gwzS', '37vFx8A1vIsrBpNlv5KSbu', '3FnF6zASCeapZDEZEWXEh0', '3OKofBUSWSKkSb9M2d2VME',
        '3eSBO1dlEmjBFrI9aWXDpy', '3h6ZzlyORFDl78dUIn9zhM', '3mgJEdJgPmjtouhddq7SDz', '3wPxSENJLY2WdaYQhMld8E',
        '3xHh5ekjTcwxzp1YMmlgTu', '3zIPYavKDr3S9elTBB2Xm0', '41JPuU92KUwcbbOzTRILCW', '4DrSZVagRCJrHXNZEZq6Dl',
        '4GCapC72RfdsDEZO3yTySe', '4Ns5OdLIZXgFAqP8rM2ak9', '4UR8XyM5iZn4SLFgNodv48', '4ereXl6uXc7CCu3WoNItRX',
        '4nA2nJsGA408bdIp2hwUbJ', '4oqbbPDhZygeUApmnII8yf', '4wDcz66gKfKkVeaE1rajm6', '4x0q4PmacquY59DiBvaCYZ',
        '511RkhtBd4ruqLOfWXfCcQ', '56LbJxigbWgc6uGrolqQl8', '5FwTXwY4TQsJfjJJIY1DGb', '5HhWCGwcnWJd2QTnOZ0xaP',
        '5NKsZvFIjmcDruIrIUp7S6', '5PLmcdhUcQLnndlncXBkW3', '5PZUtUX1Npf0phZyK35BdX', '5QPxPzo5nrMyPATFMHjCCH',
        '5SHNE4VlacA3wS6t0c6MOb', '5Z8IcSS0fYvFCts2Xivm34', '5cl1P484GnCzweSgHY3dXS', '5etv5KUlovg3l57V1Y9g6D',
        '5pJtR9Tqy3a3NrfjMCO7hS', '5xvQTG5jY6V6InJ5iGaXur', '6Fuv9Gi5oF38MgAQ9cnZSh', '6MvLbIydGnGVaVy9T0EgFc',
        '6Pr3RkmwYtJZTmbVw3aQ9u', '6WP80dOgYyUKVjFDrR6WF0', '6WhXMFGGzCa0DUCwcHFVnp', '6YNLp2ckDuwLAaRz80YnfY',
        '6jHgYHltWiQUfT0m0X97aF', '6l7VISRrTSy7lNEJDUDhvB', '6oTdx0ARlxsFPS3GmOmTOL', '6rknhO8jAvjdrXiKTB7oIB',
        '6s769bWPbEqGJw5xSfrNRw', '6vlLLd3TLxHsnKx0QzoBAj', '6z86ozjTZwOS7YiiPMAv5T', '70XP17NtabFqZPmiwAIYLw',
        '72vPB4OUttkwVLM3yk6iLY', '73Ypj3Y4eFn9Oi5DR6BaHS', '7p4SY66GqvDg02VMLhEty7', '7ptJgVQvnJZzq3il3mQQDM',
        '7zs3u26nSvkktXxCUoU3tx'
    }

    print(f"Total bogus tracks identified for removal: {len(BOGUS_TRACK_IDS)}")

    # Create mask for clean tracks based on Spotify ID
    clean_mask = (~df['id'].isin(BOGUS_TRACK_IDS)).to_numpy()
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
