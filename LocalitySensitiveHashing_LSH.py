"""
Basic Python implementation of *Locality Sensitive Hashing (LSH)* for cosine similarity,
using random hyperplanes to hash vectors. 
This is commonly used for approximate nearest neighbor search in high-dimensional spaces.
"""


"""
✅ Overview:
Uses random hyperplanes to create hash functions.

Vectors hashed into buckets based on the sign of their dot product with hyperplanes.

Similar vectors (in cosine space) will likely fall into the same bucket.
"""

import numpy as np
from collections import defaultdict

class LSH:
    def __init__(self, num_hashes=10, dim=100):
        """
        num_hashes: Number of hash functions (hyperplanes).
        dim: Dimensionality of input vectors.
        """
        self.num_hashes = num_hashes
        self.dim = dim
        self.hyperplanes = np.random.randn(num_hashes, dim)  # Random hyperplanes
        self.buckets = defaultdict(list)

    def _hash(self, vector):
        """
        Generate a hash for a vector using random hyperplanes.
        """
        projections = np.dot(self.hyperplanes, vector)
        return tuple(projections >= 0)  # Binary hash: True/False for each hyperplane

    def insert(self, vec_id, vector):
        """
        Insert a vector into a bucket based on its hash.
        vec_id: A unique identifier for the vector.
        """
        h = self._hash(vector)
        self.buckets[h].append((vec_id, vector))

    def query(self, vector):
        """
        Query similar vectors by returning the vectors in the same bucket.
        """
        h = self._hash(vector)
        return self.buckets.get(h, [])

# ----------------------------
# 🔬 Example usage
# ----------------------------
if __name__ == "__main__":
    np.random.seed(42)

    lsh = LSH(num_hashes=8, dim=5)

    # Sample 5D vectors
    vectors = {
        "A": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "B": np.array([0.11, 0.29, 0.52, 0.72, 0.91]),
        "C": np.array([-0.5, 0.2, -0.3, 0.7, -0.6]),
    }

    # Insert vectors
    for name, vec in vectors.items():
        lsh.insert(name, vec)

    # Query
    query_vector = np.array([0.1, 0.3, 0.5, 0.71, 0.88])
    results = lsh.query(query_vector)

    print("Query Results:")
    for vec_id, vec in results:
        print(f"{vec_id}: {vec}")


