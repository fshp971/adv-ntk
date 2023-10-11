import jax
import jax.numpy as jnp


def adv_batched_gdmse_ensemble(kernel_fn, x_train, y_train, batch_size, ntk=None, try_parallel=False):
    num = len(y_train)

    if try_parallel:
        n_devices = jax.device_count()
    else:
        n_devices = 1

    def _original_ktd_fn(x1, x2):
        ks = kernel_fn(x1, x2, get=("ntk",))
        return ks.ntk

    if n_devices == 1:
        ktd_fn = jax.jit(_original_ktd_fn)
    else:
        ktd_fn = jax.pmap(_original_ktd_fn, in_axes=(None, 0,))

    def hstack_ktd_fn(x1, x2):
        ntkxx = []
        for i in range(0, num, batch_size * n_devices):
            ii = min(i + batch_size * n_devices, num)

            if n_devices == 1:
                tx2 = x_train[i : ii]
                ntkxx.append(ktd_fn(x1, tx2))

            else:
                nbs = (ii-i) // batch_size
                ri = i + nbs * batch_size
                tx2 = x_train[i : ri]
                tx2 = tx2.reshape(nbs, batch_size, *tx2.shape[1:])
                ntkxx.append(jnp.hstack( ktd_fn(x1, tx2) ))

                if ri < ii:
                    tx2 = x_train[ri : ii]
                    tx2 = jnp.expand_dims(tx2, axis=0)
                    ntk.append(ktd_fn(x1, tx2).squeeze(0))

        return jnp.hstack(ntkxx)

    if ntk is None:
        ntk = []
        for i in range(0, num, batch_size):
            ii = min(i + batch_size, num)
            ntk.append(hstack_ktd_fn(x_train[i:ii], x_train))
        ntk = jnp.vstack(ntk)

    eva, evc = jnp.linalg.eigh(ntk)
    inv_ntk = jnp.einsum("ij,j,kj->ik", evc, 1/eva, evc)
    del eva, evc

    @jax.jit
    def aug_pred_mat_fn(aug):
        aug = jnp.abs(aug)
        mat = -jnp.einsum("ik,k->ik", ntk, aug)
        expm1_mat = jax.scipy.linalg.expm(mat) - jnp.identity(len(x_train), dtype=ntk.dtype)
        pred_mat = jnp.matmul(inv_ntk, jnp.matmul(-expm1_mat, y_train))
        return pred_mat

    def predict_fn(aug, x_test):
        pred_mat = aug_pred_mat_fn(aug)
        ktd = hstack_ktd_fn(x_test, x_train)
        return jnp.matmul(ktd, pred_mat)

    def _lossp(aug, ktd, y_test):
        pmat = aug_pred_mat_fn(aug)
        mean = jnp.matmul(ktd, pmat)
        return 0.5 / len(y_test) * ((mean-y_test)**2).sum()

    _gradp = jax.jit(jax.grad(_lossp, argnums=0))

    def gradp_fn(aug, xt, yt):
        ktd = hstack_ktd_fn(xt, x_train)
        return _gradp(aug, ktd, yt)

    def build_predx_gradx(aug):
        pred_mat = aug_pred_mat_fn(aug)

        def predx_fn(x_test):
            ktd = hstack_ktd_fn(x_test, x_train)
            return jnp.matmul(ktd, pred_mat)

        def weighted_loss_fn(x1, x2, pmat, weight):
            return (jnp.matmul(_original_ktd_fn(x1, x2), pmat) * weight).sum()
        weighted_grad_fn = jax.grad(weighted_loss_fn, argnums=0)
        if n_devices == 1:
            weighted_grad_fn = jax.jit(weighted_grad_fn)
        else:
            weighted_grad_fn = jax.pmap(weighted_grad_fn, in_axes=(None, 0, 0, None,))

        def gradx_fn(x_test, y_test):
            weight = predx_fn(x_test) - y_test
            gd = 0
            for i in range(0, num, batch_size * n_devices):
                ii = min(i + batch_size * n_devices, num)

                if n_devices == 1:
                    tx = x_train[i : ii]
                    pmat = pred_mat[i : ii]
                    gd += weighted_grad_fn(x_test, tx, pmat, weight)

                else:
                    nbs = (ii-i) // batch_size
                    ri = i + nbs * batch_size
                    tx = x_train[i : ri]
                    tx = tx.reshape(nbs, batch_size, *tx.shape[1:])
                    pmat = pred_mat[i : ri]
                    pmat = pmat.reshape(nbs, batch_size, *pmat.shape[1:])
                    gd += weighted_grad_fn(x_test, tx, pmat, weight).sum(axis=0)

                    if ri < ii:
                        tx = x_train[ri : ii]
                        tx = jnp.expand_dims(tx, axis=0)
                        pmat = pmat[ri : ii]
                        pmat = jnp.expand_dims(pmat, axis=0)
                        gd += weighted_grad_fn(x_test, tx, pmat, weight).squeeze(0)
            return gd

        return predx_fn, gradx_fn

    return predict_fn, gradp_fn, build_predx_gradx, ntk
    # return predict_fn, gradp_fn, build_predx_gradx
