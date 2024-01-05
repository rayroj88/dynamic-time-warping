import numpy as np

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def dtw2(sequence1, sequence2):
    """
    Compute the Dynamic Time Warping (DTW) distance between sequence1 and the best matching subsequence in sequence2.
    Also find the start and end frame of this subsequence in sequence2.

    Parameters:
    sequence1 (np.array): A 2D numpy array representing the first sequence.
    sequence2 (np.array): A 2D numpy array representing the second sequence.

    Returns:
    float: The DTW distance between the two sequences.
    int: The start frame of the best matching subsequence in sequence2.
    int: The end frame of the best matching subsequence in sequence2.
    """
    
    # TODO: Implement the DTW algorithm, and return the correct distance, start frame, and end frame.

    len_seq1 = len(sequence1[0])
    len_seq2 = len(sequence2[0])

    # Handle empty sequences
    if len_seq1 == 0 or len_seq2 == 0:
        return np.inf, 0, 0

    cost_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))
    start_frame_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1), dtype=int)
    end_frame_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1), dtype=int)

    for i in range(len_seq1 + 1):
        cost_matrix[i, 0] = np.inf
    for j in range(len_seq2 + 1):
        cost_matrix[0, j] = np.inf

    cost_matrix[0, 0] = 0

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            cost = np.linalg.norm(sequence1[:, i - 1] - sequence2[:, j - 1])
            min_prev = min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])

            cost_matrix[i, j] = cost + min_prev

            if min_prev == cost_matrix[i - 1, j]:
                start_frame_matrix[i, j] = start_frame_matrix[i - 1, j]
                end_frame_matrix[i, j] = end_frame_matrix[i - 1, j]
            elif min_prev == cost_matrix[i, j - 1]:
                start_frame_matrix[i, j] = start_frame_matrix[i, j - 1]
                end_frame_matrix[i, j] = end_frame_matrix[i, j - 1]
            else:  # min_prev == cost_matrix[i - 1, j - 1]
                start_frame_matrix[i, j] = start_frame_matrix[i - 1, j - 1]
                end_frame_matrix[i, j] = end_frame_matrix[i - 1, j - 1]

            if cost_matrix[i, j] < cost_matrix[i, end_frame_matrix[i, j]]:
                end_frame_matrix[i, j] = j

    min_distance = cost_matrix[len_seq1, 1]
    start_frame = start_frame_matrix[len_seq1, 0]
    end_frame = end_frame_matrix[len_seq1, 0]

    for j in range(2, len_seq2 + 1):
        if cost_matrix[len_seq1, j] < min_distance:
            min_distance = cost_matrix[len_seq1, j]
            end_frame = j

    start_frame = start_frame_matrix[len_seq1, end_frame]
    start_frame -= 1  # Adjust for 0-based indexing
    end_frame -= 1    # Adjust for 0-based indexing

    return min_distance, start_frame, end_frame