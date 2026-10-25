"""Observe JAX tracing/compilation boundaries on CPU or available device.
Requires JAX; reports timings for the user's environment only.
"""
import time
import jax
import jax.numpy as jnp

@jax.jit
def loss(w, x, y):
    error = x @ w - y
    return jnp.mean(error * error)

w = jnp.ones((4,))
x = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 32
y = jnp.ones((8,))
for label in ("first call", "subsequent call"):
    start = time.perf_counter()
    result = loss(w, x, y)
    result.block_until_ready()
    print(label, "seconds:", time.perf_counter() - start)
print("loss:", float(result), "gradient shape:", jax.grad(loss)(w, x, y).shape)
print(jax.make_jaxpr(loss)(w, x, y))
