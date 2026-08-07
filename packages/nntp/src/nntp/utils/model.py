import orbax.checkpoint as ocp
import jax
import flax.nnx as nnx


def restore_model_from_checkpoint(
    checkpoint_path: str,
    model,
    step: int,
) -> nnx.Module:
    # ---- restore model / optimizer ----
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef_model, abstract_model_state = nnx.split(abstract_model)

    # change sharding info to single device sharding
    abstract_model_state = jax.tree_util.tree_map(
        lambda x: x.update(sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])),
        abstract_model_state,
    )

    # restore checkpoint
    mngr = ocp.CheckpointManager(checkpoint_path, options=ocp.CheckpointManagerOptions())
    restored = mngr.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(abstract_model_state),
        ),
    )
    return nnx.merge(graphdef_model, nnx.State(restored.model, _copy=True))


def override_model_state(model, state):
    graphdef_model, _ = nnx.split(nnx.eval_shape(lambda: model))
    return nnx.merge(graphdef_model, state)


def model_reconstruct(model, state, X, Y, M, evaluator, loss_fn):
    new_model = override_model_state(model, state)
    y = new_model.forward(X)
    return {"loss": loss_fn(y, Y, M), **evaluator(y, Y, M)}
