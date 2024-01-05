import numpy as np

def dtw(sequence1, sequence2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.

    Parameters:
    sequence1 (np.array): A 2D numpy array representing the first sequence.
    sequence2 (np.array): A 2D numpy array representing the second sequence.

    Returns:
    float: The DTW distance between the two sequences.
    """
    
    # TODO: Implement the DTW algorithm, and return the correct distance.

    # Get the lengths of the input sequences
    len_seq1 = len(sequence1[0])
    len_seq2 = len(sequence2[0])
    
    # Create a cost matrix initialized with zeros
    cost_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))

    # Initialize the first row and column of the cost matrix
    for i in range(len_seq1 + 1):
        cost_matrix[i, 0] = np.inf
    for j in range(len_seq2 + 1):
        cost_matrix[0, j] = np.inf
    
    cost_matrix[0, 0] = 0

    # Compute the DTW distance using dynamic programming
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            cost = np.linalg.norm(sequence1[:, i - 1] - sequence2[:, j - 1])
            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])

    # Return the DTW distance
    return cost_matrix[len_seq1, len_seq2]
