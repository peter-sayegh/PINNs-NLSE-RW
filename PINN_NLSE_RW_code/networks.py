import jax, jax.numpy as jnp

import flax, flax.linen as nn

from typing import Sequence

from params_init import _uniform_glorot_init
from params_init import _factorized_glorot_init


#-Define available activation functions------------------------------

activation_fn = {
    'sin': jnp.sin,
    'tanh': nn.tanh,
    'gelu': nn.gelu,
    'swish': nn.swish,
    'softplus': nn.softplus
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]
    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


#-Define basic 'dense' layer-----------------------------------------

class Dense(nn.Module):
    features: int
    factorization: bool

    def setup(self):
        self.kernel_init = _factorized_glorot_init if self.factorization else _uniform_glorot_init

    @nn.compact
    def __call__(self, x):
        
        if self.factorization:
            V, s = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))  
            bias = self.param('bias', lambda key, shape: jnp.zeros(shape), (self.features,))

            return x @ (V * s) + bias

        else:
            kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
            bias = self.param('bias', lambda key, shape: jnp.zeros(shape), (self.features,))  

            return x @ kernel + bias


#-Standard MLP supporting random weight factorization----------------
        
class MLP(nn.Module):
    features: Sequence[int]
    activation: str = 'tanh'
    factorization: bool = False

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        *hidden_features, out_dim = self.features

        for features in hidden_features:
            x = Dense(features, factorization=self.factorization)(x)
            x = self.activation_fn(x)
        
        return Dense(out_dim, factorization=self.factorization)(x)