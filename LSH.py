import hashlib
import numpy as np
from sklearn.neighbors import NearestNeighbors

def minhash(shingles, num_hashes=200):
    """Generate minhash signature for a set of shingles."""
    signature = []
    for i in range(num_hashes):
        # Generate hash using md5 and convert to integer for compatibility with NearestNeighbors
        min_hash = min([(int(hashlib.md5((str(i) + shingle).encode()).hexdigest(), 16)) for shingle in shingles])
        signature.append(min_hash)
    return signature

def lsh(strings, threshold=0.8, num_hashes=200):
    """Determine if two strings are related using Locality Sensitive Hashing."""
    k = 3  # Shingle length (size of substrings)
    shingles_list = []
    
    for string in strings:
        shingles = set([string[i:i+k] for i in range(len(string)-k+1)])
        shingles_list.append(shingles)
    
    # Step 2: Create minhash signatures for each string
    signatures = np.array([minhash(shingles, num_hashes) for shingles in shingles_list])
    
    # Step 3: LSH - Check if any pair of strings are similar (Jaccard similarity)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
    
    # Convert the minhash signatures into a numeric matrix (from list of integers)
    nbrs.fit(signatures)

    # Step 4: Compare the signatures
    pairs = []
    for i in range(len(strings)):
        distances, indices = nbrs.kneighbors([signatures[i]])
        # Check if the distance is below a certain threshold (this implies similarity)
        for idx, distance in zip(indices[0], distances[0]):
            if i != idx and distance < threshold:
                pairs.append((strings[i], strings[idx], distance))
    
    return pairs

# Example usage
# strings = [
#     "nankai university",
#     "nankai university",
# ]

# similar_pairs = lsh(strings, threshold=0.5)
# print(similar_pairs)
# for s1, s2, dist in similar_pairs:
#     print(f"Related Strings: '{s1}' and '{s2}' with distance: {dist}")
