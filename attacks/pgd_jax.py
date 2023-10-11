import functools
import jax
import jax.numpy as jnp


def PGDAttack(radius, steps, step_size, norm_type):
    radius = radius / 255.
    step_size = step_size / 255.

    def _clip_fn(adv_x, x):
        adv_x = adv_x - x
        if norm_type == "l-infty":
            adv_x = jnp.clip(adv_x, -radius, radius)
        else:
            if norm_type == "l2":
                norm = jnp.sqrt(jnp.sum(adv_x**2, axis=jnp.arange(1,adv_x.ndim), keepdims=True))
            elif norm_type == "l1":
                norm = jnp.sum(jnp.abs(adv_x), axis=jnp.arange(1,adv_x.ndim), keepdims=True)
            adv_x /= (norm + 1e-10)
            adv_x *= jnp.clip(norm, a_max=radius)
        adv_x = adv_x + x
        adv_x = jnp.clip(adv_x, -0.5, 0.5)
        return adv_x

    def rand_init_fn(x, key):
        if norm_type == 'l-infty':
            adv_x = (2 * radius) * (jax.random.uniform(key, shape=x.shape, dtype=x.dtype) - 0.5)
        else:
            adv_x = (2 * radius / steps) * (jax.random.uniform(key, shape=x.shape, dtype=x.dtype) - 0.5)
        adv_x = adv_x + x
        adv_x = _clip_fn(adv_x, x)
        return adv_x

    def perturb_fn(grad_fn, x, y):
        adv_x = x
        for step in range(steps):
            gd = grad_fn(x, y)
            if norm_type == "l-infty":
                gd = jnp.sign(gd)
            else:
                if norm_type == "l2":
                    gd_norm = jnp.sqrt(jnp.sum(gd**2, axis=jnp.arange(1,gd.ndim), keepdims=True))
                elif norm_type == "l1":
                    gd_norm = jnp.sum(jnp.abs(gd), axis=jnp.arange(1,gd.ndim), keepdims=True)
                gd = gd / (gd_norm + 1e-10)
            gd = gd * step_size
            adv_x = _clip_fn(adv_x + gd, x)
        return adv_x

    return rand_init_fn, perturb_fn
