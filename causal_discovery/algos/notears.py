import jax
import optax
import jax.numpy as jnp

from typing import List
from tqdm import trange


class NoTears:
    """
    An implementation of the NoTEARS algorithm for causal discovery using gradient descent.

    Paper: https://arxiv.org/abs/1803.01422
    """
    # TODO: need to refactor
    # TODO: support categorical variables
    
    def __init__(self, rho: float, alpha: float, l1_reg: float, lr: float=1e-3):
        self.rho = rho
        self.alpha = alpha
        self.l1_reg = l1_reg
        self.opt = optax.adam(lr)
        self.W = None
        
        self.loss_fn = jax.jit(self.loss_fn)
        
    def init_params(self, n: int) -> jnp.ndarray:
        """
        Initialize the parameters.
        
        Args:
            n: the number of features.
        """
        
        return jnp.zeros((n, n))
    
    def h(self, W: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the h(W) term in (16).
        
        Args:
            W: the weight matrix.
        """
        
        return jnp.trace(jax.scipy.linalg.expm(W * W)) - W.shape[0]
    
    def loss_fn(self, W: jnp.ndarray, X: jnp.ndarray, rho: float, alpha: float) -> jnp.ndarray:
        """
        Calculate the loss (ECP) according to (15), (16).
        
        Args:
            W: the weight matrix.
            X: the data matrix.
            rho: the regularization parameter.
            alpha: the regularization parameter.
            
        Returns:
            ECP loss
        """
        
        n, d = X.shape
        
        W = W - jnp.diag(jnp.diag(W))
        
        mse = 0.5 / n * jnp.square(jnp.linalg.norm(X - X @ W))
        reg = self.l1_reg * jnp.linalg.norm(W, ord=1)
        h = self.h(W)
        rho_term = 0.5 * rho * h * h
        alpha_term = alpha * h
        
        return mse + reg + rho_term + alpha_term
    
    def learn(self, X: jnp.ndarray, n_outer_iter: int=20, n_inner_iter: int=100) -> List[float]:
        """
        Learn the weight matrix.
        
        Args:
            X: the data matrix.
            n_outer_iter: the number of outer iterations.
            n_inner_iter: the number of inner iterations.
        """
        
        n = X.shape[1]
        self.W = self.init_params(n)
        opt_state = self.opt.init(self.W)
        loss_history = []
        h = jnp.inf
        
        for _ in trange(n_outer_iter):
            while self.rho < 1e6:
                for _ in range(n_inner_iter):
                    loss, grad = jax.value_and_grad(self.loss_fn)(self.W, X, self.rho, self.alpha)
                    updates, opt_state = self.opt.update(grad, opt_state)
                    self.W = optax.apply_updates(self.W, updates)
                    loss_history.append(float(loss))

                new_h = self.h(self.W)                
                if new_h > 0.25 * h:
                    self.rho *= 10
                else:
                    break
                
            h = new_h
            self.alpha += self.rho * h
        
        return loss_history
