import torch
import torch.nn.functional as F

def find_matches_cos_similarity(spot_vectors, query_vectors, top_k=1):
    """
    Find the top-k most similar scRNA-Seq cells for each Xenium cell using cosine similarity.

    Args:
        spot_vectors (array): 2D array of gene expression of scRNA-Seq cells.
        query_vectors (array): 2D array of gene expression of Xenium cells.
        top_k (int): Number of top matches to return.

    Returns:
        tuple: Indices and similarity values of top-k matches.
    """

    # Use PyTorch functionalities for efficiency and scalability
    # Normalize the vectors
    spot_vectors = F.normalize(torch.tensor(spot_vectors, dtype=torch.float32), p=2, dim=-1).cuda()
    query_vectors = F.normalize(torch.tensor(query_vectors, dtype=torch.float32), p=2, dim=-1).cuda()

    # Compute dot product (cosine similarity because vectors are first normalized to unit norm)
    dot_similarity = query_vectors @ spot_vectors.T

    # Find the top_k similar spots for each query
    values, indices = torch.topk(dot_similarity, k=top_k, dim=-1)
    return indices.cpu().numpy(), values.cpu().numpy()