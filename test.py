import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.interpolate import BSpline

from bspline.matrix_evaluation import matrix_bspline_evaluation_at_times

import spline_opt_tools


def main():
    x0 = np.array(
        [
            -6.99550637,
            -8.95872567,
            -4.47336006,
            -5.06316688,
            -5.11105337,
            -0.78860679,
            -3.19745253,
            3.19381798,
            0.79036742,
            5.04575115,
            5.06051916,
            4.51374057,
            8.96755593,
            6.89928658,
        ]
    )
    controlPoints = x0.reshape(-1, 2)
    numControlPoints = len(controlPoints)
    t0 = 0.0
    tf = 4.0
    splineOrder = 3
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        t0, tf, numControlPoints, splineOrder
    )
    numSamplesPerInterval = 10

    # one call to compile jit
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )
    vel = spline_opt_tools.evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    startTime = time.time()
    pos = spline_opt_tools.evaluate_spline(
        controlPoints, knotPoints, numSamplesPerInterval
    )
    vel = spline_opt_tools.evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    print("Time taken for evaluate_spline: ", time.time() - startTime)

    scipySpline = BSpline(knotPoints, controlPoints, splineOrder)
    # test scipy spline
    t = np.linspace(
        t0, tf, (numControlPoints - splineOrder) * numSamplesPerInterval + 1
    )
    startTime = time.time()
    scipySplinePos = scipySpline(t)
    scipySplineVel = scipySpline.derivative(1)(t)
    print("Time taken for scipy spline: ", time.time() - startTime)

    # p    # print(pos)
    #
    print(np.isclose(pos, scipySplinePos).all())
    print(np.isclose(vel, scipySplineVel, atol=1e-8).all())
    # print(vel)
    # print(scipySplineVel)
    fig, ax = plt.subplots()
    plt.plot(pos[:, 0], pos[:, 1])
    plt.plot(scipySplinePos[:, 0], scipySplinePos[:, 1])
    plt.plot(vel[:, 0], vel[:, 1], c="r")
    plt.plot(scipySplineVel[:, 0], scipySplineVel[:, 1], c="g")
    plt.show()


def new_test():
    x0 = np.array(
        [
            -6.99550637,
            -8.95872567,
            -4.47336006,
            -5.06316688,
            -5.11105337,
            -0.78860679,
            -3.19745253,
            3.19381798,
            0.79036742,
            5.04575115,
            5.06051916,
            4.51374057,
            8.96755593,
            6.89928658,
        ]
    )
    controlPoints = x0.reshape(-1, 2)
    numControlPoints = len(controlPoints)
    t0 = 0.0
    tf = 4.0
    splineOrder = 3
    knotPoints = spline_opt_tools.create_unclamped_knot_points(
        t0, tf, numControlPoints, splineOrder
    )
    scipySpline = BSpline(knotPoints, controlPoints, splineOrder)

    t = np.linspace(t0, 4.0, 1000)
    # t = np.array([tf])

    startTime = time.time()
    scipyPos = scipySpline(t)
    print("Time taken for scipy spline: ", time.time() - startTime)
    pos = matrix_bspline_evaluation_at_times(t, controlPoints, knotPoints, splineOrder)
    startTime = time.time()
    pos = matrix_bspline_evaluation_at_times(t, controlPoints, knotPoints, splineOrder)
    print("Time taken for evaluate_spline: ", time.time() - startTime)

    print(pos[-1])
    print(scipyPos[-1])

    print(np.isclose(pos, scipyPos).all())


if __name__ == "__main__":
    new_test()
