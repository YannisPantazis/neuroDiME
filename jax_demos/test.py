import flax.linen as nn
from flax.training.train_state import TrainState
import jax, jax.numpy as jnp
import optax

x = jnp.ones((1, 2))
y = jnp.ones((1, 2))
model = nn.Dense(2)
variables = model.init(jax.random.key(0), x)
tx = optax.adam(1e-3)

print(jax.tree_map(lambda x: x.shape, variables)) # Check the parameters


test = nn.tabulate(model, jax.random.key(0))
print(test(x))

state = TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)

def loss_fn(params, x, y):
  predictions = state.apply_fn({'params': params}, x)
  print(predictions.shape)
  loss = optax.l2_loss(predictions=predictions, targets=y).mean()
  return loss
# loss_fn(state.params, x, y)

print('Before')
print(state.params)

grads = jax.grad(loss_fn)(state.params, x, y)
state = state.apply_gradients(grads=grads)
print('After')
print(state.params)
# loss_fn(state.params, x, y)
