import numpy as np

def bass_model(t, p, q, M):
    """
    Mathematical implementation of the Bass Diffusion Model.
    
    Parameters:
    t : array - time periods
    p : float - coefficient of innovation
    q : float - coefficient of imitation
    M : int   - total market potential
    
    Returns: Predicted annual adopters for each period t.
    """
    S_pred = np.zeros(len(t))
    A = np.zeros(len(t)) # Cumulative adopters
    
    for i in range(len(t)):
        prev_A = A[i-1] if i > 0 else 0
        S_pred[i] = p * M + (q - p) * prev_A - (q / M) * (prev_A**2)
        A[i] = prev_A + S_pred[i]
    return S_pred
