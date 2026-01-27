"""
Msty Admin MCP - Embedding Visualization Tools

Tools for visualizing and analyzing embeddings from Knowledge Stacks,
including clustering and similarity analysis.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .paths import get_msty_paths
from .database import get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# EMBEDDING DATA ACCESS
# ============================================================================

def get_embeddings_from_stack(stack_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get embeddings from a knowledge stack."""
    embeddings = []
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()

            # Look for embedding tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND (
                    name LIKE '%embedding%' OR
                    name LIKE '%vector%' OR
                    name LIKE '%chunk%'
                )
            """)
            tables = [t[0] for t in cursor.fetchall()]

            for table in tables:
                try:
                    cursor.execute(f"""
                        SELECT * FROM {table}
                        WHERE stack_id = ? OR knowledge_stack_id = ?
                        LIMIT ?
                    """, (stack_id, stack_id, limit))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        embeddings.append(dict(zip(columns, row)))
                except:
                    pass

            conn.close()
    except Exception as e:
        logger.debug(f"Embedding query error: {e}")

    return embeddings


# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

def visualize_embeddings(
    stack_id: str,
    method: str = "pca",
    dimensions: int = 2
) -> Dict[str, Any]:
    """
    Generate visualization data for embeddings.

    Note: Full visualization requires numpy/sklearn. This provides
    a simplified version using basic statistics.

    Args:
        stack_id: Knowledge stack to visualize
        method: Dimensionality reduction method (pca, tsne, umap)
        dimensions: Output dimensions (2 or 3)

    Returns:
        Dict with visualization data
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "method": method,
        "dimensions": dimensions,
        "visualization_data": [],
    }

    embeddings = get_embeddings_from_stack(stack_id)

    if not embeddings:
        result["error"] = f"No embeddings found for stack '{stack_id}'"
        result["suggestion"] = "Ensure the knowledge stack has been processed with an embedding model"
        return result

    result["embedding_count"] = len(embeddings)

    # Extract vectors
    vectors = []
    metadata = []
    for emb in embeddings:
        vector = emb.get("embedding") or emb.get("vector")
        if vector:
            # Handle string representation of vector
            if isinstance(vector, str):
                try:
                    vector = json.loads(vector)
                except:
                    continue
            if isinstance(vector, list):
                vectors.append(vector)
                metadata.append({
                    "id": emb.get("id"),
                    "chunk_id": emb.get("chunk_id"),
                    "text_preview": (emb.get("text", "") or emb.get("content", ""))[:100],
                    "source": emb.get("source") or emb.get("document_id")
                })

    if not vectors:
        result["error"] = "Could not extract embedding vectors"
        return result

    result["vectors_found"] = len(vectors)
    result["vector_dimensions"] = len(vectors[0]) if vectors else 0

    # Simple dimensionality reduction (placeholder - real PCA would need numpy)
    # We'll provide pseudo-2D coordinates based on vector statistics
    visualization_data = []
    for i, (vector, meta) in enumerate(zip(vectors, metadata)):
        # Use simple statistics as proxy coordinates
        x = sum(vector[:len(vector)//2]) / (len(vector)//2) if vector else 0
        y = sum(vector[len(vector)//2:]) / (len(vector)//2) if vector else 0

        # Normalize
        magnitude = math.sqrt(x*x + y*y)
        if magnitude > 0:
            x, y = x / magnitude, y / magnitude

        point = {
            "x": round(x, 4),
            "y": round(y, 4),
            "id": meta.get("id"),
            "label": meta.get("text_preview", "")[:50],
            "source": meta.get("source")
        }

        if dimensions == 3:
            # Add z dimension from middle portion
            z = sum(vector[len(vector)//3:2*len(vector)//3]) / (len(vector)//3) if vector else 0
            point["z"] = round(z / (magnitude if magnitude > 0 else 1), 4)

        visualization_data.append(point)

    result["visualization_data"] = visualization_data
    result["note"] = "Simplified visualization. For production use, integrate with numpy/sklearn for proper PCA/t-SNE."

    return result


# ============================================================================
# EMBEDDING CLUSTERING
# ============================================================================

def cluster_embeddings(
    stack_id: str,
    n_clusters: int = 5,
    method: str = "kmeans"
) -> Dict[str, Any]:
    """
    Cluster embeddings from a knowledge stack.

    Args:
        stack_id: Knowledge stack to cluster
        n_clusters: Number of clusters
        method: Clustering method (kmeans, hierarchical)

    Returns:
        Dict with clustering results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "method": method,
        "n_clusters": n_clusters,
        "clusters": [],
    }

    embeddings = get_embeddings_from_stack(stack_id)

    if not embeddings:
        result["error"] = f"No embeddings found for stack '{stack_id}'"
        return result

    # Extract vectors
    vectors = []
    metadata = []
    for emb in embeddings:
        vector = emb.get("embedding") or emb.get("vector")
        if vector:
            if isinstance(vector, str):
                try:
                    vector = json.loads(vector)
                except:
                    continue
            if isinstance(vector, list):
                vectors.append(vector)
                metadata.append({
                    "id": emb.get("id"),
                    "text": (emb.get("text", "") or emb.get("content", ""))[:200],
                    "source": emb.get("source") or emb.get("document_id")
                })

    if len(vectors) < n_clusters:
        result["error"] = f"Not enough vectors ({len(vectors)}) for {n_clusters} clusters"
        return result

    # Simple k-means-like clustering (simplified without numpy)
    # Initialize cluster centers using first n_clusters points
    centers = vectors[:n_clusters]

    # Assign each vector to nearest center
    assignments = []
    for vector in vectors:
        min_dist = float('inf')
        best_cluster = 0
        for i, center in enumerate(centers):
            dist = sum((a - b) ** 2 for a, b in zip(vector, center))
            if dist < min_dist:
                min_dist = dist
                best_cluster = i
        assignments.append(best_cluster)

    # Group by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for i, (assignment, meta) in enumerate(zip(assignments, metadata)):
        clusters[assignment].append({
            "index": i,
            **meta
        })

    # Format results
    for cluster_id, members in clusters.items():
        cluster_info = {
            "cluster_id": cluster_id,
            "member_count": len(members),
            "members": members[:10],  # Limit members shown
            "sample_texts": [m.get("text", "")[:100] for m in members[:3]]
        }
        result["clusters"].append(cluster_info)

    result["total_items"] = len(vectors)
    result["cluster_sizes"] = {
        f"cluster_{i}": len(clusters[i])
        for i in range(n_clusters)
    }

    return result


# ============================================================================
# EMBEDDING SIMILARITY
# ============================================================================

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(v1) != len(v2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def compare_embeddings(
    stack_id: str,
    query_text: Optional[str] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Compare embeddings and find similar documents.

    Args:
        stack_id: Knowledge stack to search
        query_text: Text to find similar documents for (uses first doc if None)
        top_k: Number of similar documents to return

    Returns:
        Dict with similarity results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "top_k": top_k,
        "similar_documents": [],
    }

    embeddings = get_embeddings_from_stack(stack_id, limit=500)

    if not embeddings:
        result["error"] = f"No embeddings found for stack '{stack_id}'"
        return result

    # Extract vectors
    vectors = []
    metadata = []
    for emb in embeddings:
        vector = emb.get("embedding") or emb.get("vector")
        text = emb.get("text", "") or emb.get("content", "")
        if vector:
            if isinstance(vector, str):
                try:
                    vector = json.loads(vector)
                except:
                    continue
            if isinstance(vector, list):
                vectors.append(vector)
                metadata.append({
                    "id": emb.get("id"),
                    "text": text,
                    "source": emb.get("source") or emb.get("document_id")
                })

    if len(vectors) < 2:
        result["error"] = "Need at least 2 embeddings for comparison"
        return result

    # Select query vector
    query_idx = 0
    if query_text:
        # Find best matching text
        best_match = 0
        best_score = 0
        for i, meta in enumerate(metadata):
            # Simple text overlap score
            query_words = set(query_text.lower().split())
            doc_words = set(meta.get("text", "").lower().split())
            overlap = len(query_words & doc_words)
            if overlap > best_score:
                best_score = overlap
                best_match = i
        query_idx = best_match
        result["query_matched_to"] = metadata[query_idx].get("text", "")[:100]

    query_vector = vectors[query_idx]
    result["query_text"] = metadata[query_idx].get("text", "")[:200]

    # Calculate similarities
    similarities = []
    for i, (vector, meta) in enumerate(zip(vectors, metadata)):
        if i == query_idx:
            continue
        sim = cosine_similarity(query_vector, vector)
        similarities.append({
            "index": i,
            "similarity": round(sim, 4),
            **meta
        })

    # Sort by similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    result["similar_documents"] = similarities[:top_k]
    result["total_compared"] = len(similarities)

    # Statistics
    all_sims = [s["similarity"] for s in similarities]
    if all_sims:
        result["statistics"] = {
            "max_similarity": round(max(all_sims), 4),
            "min_similarity": round(min(all_sims), 4),
            "avg_similarity": round(sum(all_sims) / len(all_sims), 4),
        }

    return result


def embedding_statistics(stack_id: str) -> Dict[str, Any]:
    """
    Get statistics about embeddings in a knowledge stack.

    Args:
        stack_id: Knowledge stack to analyze

    Returns:
        Dict with embedding statistics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "statistics": {},
    }

    embeddings = get_embeddings_from_stack(stack_id, limit=1000)

    if not embeddings:
        result["error"] = f"No embeddings found for stack '{stack_id}'"
        return result

    # Extract vectors
    vectors = []
    text_lengths = []
    sources = set()

    for emb in embeddings:
        vector = emb.get("embedding") or emb.get("vector")
        text = emb.get("text", "") or emb.get("content", "")
        source = emb.get("source") or emb.get("document_id")

        if source:
            sources.add(source)
        text_lengths.append(len(text))

        if vector:
            if isinstance(vector, str):
                try:
                    vector = json.loads(vector)
                except:
                    continue
            if isinstance(vector, list):
                vectors.append(vector)

    result["statistics"] = {
        "total_embeddings": len(embeddings),
        "valid_vectors": len(vectors),
        "unique_sources": len(sources),
        "vector_dimensions": len(vectors[0]) if vectors else 0,
    }

    if text_lengths:
        result["statistics"]["text_stats"] = {
            "avg_length": round(sum(text_lengths) / len(text_lengths)),
            "min_length": min(text_lengths),
            "max_length": max(text_lengths),
            "total_characters": sum(text_lengths)
        }

    if vectors:
        # Calculate vector statistics
        magnitudes = [math.sqrt(sum(v*v for v in vec)) for vec in vectors]
        result["statistics"]["vector_stats"] = {
            "avg_magnitude": round(sum(magnitudes) / len(magnitudes), 4),
            "min_magnitude": round(min(magnitudes), 4),
            "max_magnitude": round(max(magnitudes), 4),
        }

        # Density estimate (average pairwise similarity of sample)
        sample_size = min(20, len(vectors))
        sample = vectors[:sample_size]
        pairwise_sims = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                pairwise_sims.append(cosine_similarity(sample[i], sample[j]))

        if pairwise_sims:
            result["statistics"]["density"] = {
                "avg_pairwise_similarity": round(sum(pairwise_sims) / len(pairwise_sims), 4),
                "sample_size": sample_size
            }

    return result


__all__ = [
    "get_embeddings_from_stack",
    "visualize_embeddings",
    "cluster_embeddings",
    "compare_embeddings",
    "embedding_statistics",
    "cosine_similarity",
]
