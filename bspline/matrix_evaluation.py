import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jax.scipy.special import factorial
from jax import jit
from functools import partial


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@jax.jit
def find_previous_knot_index_uniform(time, t_min, t_max, num_knots):
    dt = (t_max - t_min) / (num_knots - 1)  # knot spacing
    index = jnp.floor((time - t_min) / dt).astype(
        jnp.int64
    )  # find the index of the previous knot
    return jnp.clip(index, 0, num_knots - 2)  # clip the index to the valid range


# # starting from scratch with the matrix method
@partial(jit, static_argnums=(3,))
def matrix_bspline_evaluation(time, control_points, knot_points, spline_order, M):
    # previousKnotIndex = jnp.searchsorted(knot_points, time, side="right") - 1
    previousKnotIndex = find_previous_knot_index_uniform(
        time, knot_points[0], knot_points[-1], len(knot_points)
    )
    previousKnotIndex = jnp.clip(
        previousKnotIndex, spline_order, len(knot_points) - 2 - spline_order
    )

    previousKnot = knot_points[previousKnotIndex]
    # jax.debug.print("previous knot {x}", x=previousKnot)
    initialControlPointIndex = (previousKnotIndex - spline_order).astype(jnp.int64)
    num_dimensions = control_points.shape[1]
    P = jax.lax.dynamic_slice(
        control_points,
        (initialControlPointIndex, 0),
        (spline_order + 1, num_dimensions),
    )
    deltaT = time - previousKnot
    powers = jnp.arange(spline_order, -1, -1)  # Generate power values [order, ..., 0]
    T = (deltaT) ** powers  # Vectorized computation
    T = T.reshape((spline_order + 1, 1))
    return (P.T @ M @ T).reshape(-1)


@partial(jit, static_argnums=(3,))
def matrix_bspline_evaluation_at_times(time, control_points, knot_points, spline_order):
    M = get_M_matrix(spline_order)
    return jax.vmap(matrix_bspline_evaluation, in_axes=(0, None, None, None, None))(
        time, control_points, knot_points, spline_order, M
    )


@partial(jit, static_argnums=(4,))
def matrix_bspline_derivative_evaluation(
    time, derivative_order, control_points, knot_points, spline_order, M
):
    # previousKnotIndex = jnp.searchsorted(knot_points, time, side="right") - 1
    previousKnotIndex = find_previous_knot_index_uniform(
        time, knot_points[0], knot_points[-1], len(knot_points)
    )
    previousKnotIndex = jnp.clip(
        previousKnotIndex, spline_order, len(knot_points) - 2 - spline_order
    )

    previousKnot = knot_points[previousKnotIndex]
    # jax.debug.print("previous knot {x}", x=previousKnot)
    initialControlPointIndex = (previousKnotIndex - spline_order).astype(jnp.int64)
    num_dimensions = control_points.shape[1]
    P = jax.lax.dynamic_slice(
        control_points,
        (initialControlPointIndex, 0),
        (spline_order + 1, num_dimensions),
    )

    deltaT = time - previousKnot
    # Generate power indices: [order, order-1, ..., 0]
    powers = jnp.arange(spline_order, -1, -1, dtype=jnp.int64)

    # Compute factorial terms using JAX's optimized function
    factorials = factorial(powers, exact=False)  # Vectorized factorial computation

    # Compute the valid indices (0 ≤ i ≤ order - rth_derivative)
    valid_mask = powers >= derivative_order

    # Compute T_derivative using JAX vectorized operations
    T = jnp.where(
        valid_mask,
        (deltaT ** (powers - derivative_order))
        * (
            factorials / factorial(powers - derivative_order, exact=False)
        ),  # Safe factorial division
        0.0,  # Set invalid entries to zero
    )
    T = T.reshape((spline_order + 1, 1))
    return (P.T @ M @ T).reshape(-1)


@partial(jit, static_argnums=(4,))
def matrix_bspline_derivative_evaluation_at_times(
    time, derivative_order, control_points, knot_points, spline_order
):
    M = get_M_matrix(spline_order)
    return jax.vmap(
        matrix_bspline_derivative_evaluation, in_axes=(0, None, None, None, None, None)
    )(time, derivative_order, control_points, knot_points, spline_order, M)


def get_M_matrix(order):
    if order > 7:
        print(
            "Error: Cannot compute higher than 7th order matrix evaluation for open spline"
        )
        return None
    elif order == 2:
        M = __get_2_order_matrix()
    elif order == 3:
        M = __get_3_order_matrix()
    elif order == 4:
        M = __get_4_order_matrix()
    elif order == 5:
        M = __get_5_order_matrix()
    elif order == 6:
        M = __get_6_order_matrix()
    elif order == 7:
        M = __get_7_order_matrix()
    return M


def __get_1_order_matrix():
    M = np.array([[-1, 1], [1, 0]])
    return M


def __get_2_order_matrix():
    M = 0.5 * np.array([[1, -2, 1], [-2, 2, 1], [1, 0, 0]])
    return M


def __get_3_order_matrix():
    M = np.array([[-2, 6, -6, 2], [6, -12, 0, 8], [-6, 6, 6, 2], [2, 0, 0, 0]]) / 12
    return M


def __get_4_order_matrix():
    M = (
        np.array(
            [
                [1, -4, 6, -4, 1],
                [-4, 12, -6, -12, 11],
                [6, -12, -6, 12, 11],
                [-4, 4, 6, 4, 1],
                [1, 0, 0, 0, 0],
            ]
        )
        / 24
    )
    return M


def __get_5_order_matrix():
    M = (
        np.array(
            [
                [-1, 5, -10, 10, -5, 1],
                [5, -20, 20, 20, -50, 26],
                [-10, 30, 0, -60, 0, 66],
                [10, -20, -20, 20, 50, 26],
                [-5, 5, 10, 10, 5, 1],
                [1, 0, 0, 0, 0, 0],
            ]
        )
        / 120
    )
    return M


def __get_6_order_matrix():
    M = (
        np.array(
            [
                [1, -6, 15, -20, 15, -6, 1],
                [-6, 30, -45, -20, 135, -150, 57],
                [15, -60, 30, 160, -150, -240, 302],
                [-20, 60, 30, -160, -150, 240, 302],
                [15, -30, -45, 20, 135, 150, 57],
                [-6, 6, 15, 20, 15, 6, 1],
                [1, 0, 0, 0, 0, 0, 0],
            ]
        )
        / 720
    )
    return M


def __get_7_order_matrix():
    M = (
        np.array(
            [
                [-1, 7, -21, 35, -35, 21, -7, 1],
                [7, -42, 84, 0, -280, 504, -392, 120],
                [-21, 105, -105, -315, 665, 315, -1715, 1191],
                [35, -140, 0, 560, 0, -1680, 0, 2416],
                [-35, 105, 105, -315, -665, 315, 1715, 1191],
                [21, -42, -84, 0, 280, 504, 392, 120],
                [-7, 7, 21, 35, 35, 21, 7, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        / 5040
    )
    return M
