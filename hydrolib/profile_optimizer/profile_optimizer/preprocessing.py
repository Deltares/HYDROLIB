import math

import numpy as np
from sympy import Symbol, solve


def search_window(b_start_value, bandwidth_perc, iterations):
    """Creates an array with bottom-widths in a bandwidth around a start value"""
    max_bound_b = b_start_value * (1 + (bandwidth_perc / 100))
    min_bound_b = b_start_value * (1 - (bandwidth_perc / 100))
    b_waardes_binnen_zoekruimte = np.linspace(min_bound_b, max_bound_b, iterations)
    return b_waardes_binnen_zoekruimte


# Ideas:
# Checks with material (sand/clay/peat/etc) if talud & velocity are okay (vademecum)


def determine_v_with_manning(d, talud, b, slope, kmanning):
    """Solve the Manning equation to find the velocity given the input parameters"""
    A = calculate_area(b, d, talud)
    R = A / (b + math.sqrt(d**2 + (d * talud) ** 2) * 2)
    V = R ** (2 / 3) * kmanning * slope ** (1 / 2)
    return V


def check_QVA(
    Q_target: float,
    d: float,
    talud: float,
    b: float,
    slope: float,
    kmanning: float,
    allowed_variation=0.05,
):
    """Given an initial b (bottom width), check if the required discharge fits the profile at the desired velocity

    Following steps are done:
    - If the initial bottom width was negative, the function will start at a bottom width of 1 m.
    - Given the input parameters, checks which velocity is expected based on Manning equation.
    - By multiplying the expected velocity with the wet area (A) of the profile, a discharge is calculated (Q=V*A)
    - If the calculated Q is not within the allowed variation from the target discharge (Q_target),
      the bottom width is adjusted (with 5%) again and again, until the target is met or until 20 iterations are done.

    # NOTE: This is not great. But it is difficult to satisfy both velocity and discharge with a few static user inputs.
    # This is really intended as a starting point for a proper solution by the profile optimizer.
    # This could really be skipped in favor of expert judgement in most use-cases.
    # Use with caution and common sense.

    # NOTE: this code only checks symmetrical taluds, while assymetric is allowed in the profile optimizer.

    Args:
        Q_target: The desired discharge
        d: The desired water depth
        talud: The desired talud (slope in profile)
        b: initial bottom width
        slope: slope of the channel
        kmanning: friction of the bed as k manning value
        allowed variation: how close does the calculated Q need to be to the target? default = 0.05 (5% variation)
    """
    if b < 0:
        b = 1
        print(
            "First guess for bottom width was negative. Trying to find a positive bottom width, starting at 1 m..."
        )

    V = determine_v_with_manning(d, talud, b, slope, kmanning)
    A = calculate_area(b, d, talud)
    Q = V * A
    print(f"Initial: width: {b:.2f}, V: {V:.4f}, Q: {Q:.4f}")
    deviation_from_target = (Q - Q_target) / Q_target
    counter = 0
    while abs(deviation_from_target) > allowed_variation:
        counter += 1
        stepsize = 0.05  # %
        if deviation_from_target < -allowed_variation:
            b *= 1 + stepsize
            V = determine_v_with_manning(d, talud, b, slope, kmanning)
            A = calculate_area(b, d, talud)
            Q = V * A
            deviation_from_target = (Q - Q_target) / Q_target
            print(f"Adjustment {counter}: new width: {b:.2f}, V: {V:.4f}, Q: {Q:.4f}")
        elif deviation_from_target > allowed_variation:
            b *= 1 - stepsize
            V = determine_v_with_manning(d, talud, b, slope, kmanning)
            A = calculate_area(b, d, talud)
            Q = V * A
            deviation_from_target = (Q - Q_target) / Q_target
            print(f"Adjustment {counter}: new width: {b:.2f}, V: {V:.4f}, Q: {Q:.4f}")

        if counter == 30:
            print(
                "Failed to find suitable initial bottom width in 30 tries, please check if your inputs are correct "
                "and use the returned bottom width with caution."
            )
            return b
    else:
        return b


def bottom_width(kmanning, slope, talud, depth, V_target):
    """Solves Manning equation for unknown bottom width

    Args:
        kmanning: bed friction value (K manning)
        slope: channel slope (m/m)
        talud: talud, slopes of the profile
        depth: water depth
        V_target: desired velocity

    Returns:
        Bottom width
    """
    R_23 = V_target / (kmanning * slope ** (1 / 2))
    b = Symbol("b")
    A = calculate_area(b, depth, talud)
    P = b + (2 * math.sqrt(depth**2 + (depth * talud) ** 2))
    eq = (A / P) ** (2 / 3) - R_23
    return solve(eq, b)


def calculate_area(width, depth, talud):
    return (width * depth) + (depth * depth * talud)
