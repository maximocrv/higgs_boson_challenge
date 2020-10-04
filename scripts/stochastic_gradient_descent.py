# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    sgd =  -1/y.shape[0] * tx.transpose() @ (y - tx @ w)
    
    return sgd


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    losses = []
    ws = []
    
    w = initial_w
    
    for i, (batch_y, batch_tx) in enumerate(batch_iter(y, tx, batch_size = batch_size, num_batches = max_iters)):
        grad = compute_stoch_gradient(batch_y, batch_tx, w)
        loss = compute_loss(batch_y, batch_tx, w)
        
        w = w - gamma*grad
        
        losses.append(loss)
        ws.append(w)
        
#         print(f'Gradient Descent ({i}/{max_iters-1}): loss={loss}, w0={w[0]}, w1={w[1]}')
    
    return losses, ws