def derivative(fpoly):
    """derivative(fpoly)
    Given a set of polynomial coefficients from highest order to x^0,
    compute the derivative polynomail.  We assume zero coefficients
    are present in the coefficient list/tuple.

    Returns polynomial coefficients for the derivative polynomial.
    Example:
    derivative((3,4,5))  # 3 * x**2 + 4 * x**1 + 5 * x**0
    returns:  [6, 4]     # 6 * x**1 + 4 * x**0
    """
    returnCoefficients = []
    fpoly = fpoly[::-1]
    index = 0
    for i in fpoly:
        returnCoefficients.append(i * index)
        index = index + 1
    returnCoefficients.pop(0)
    returnCoefficients.reverse()
    return returnCoefficients


    
def polyval(fpoly, x):
    """polyval(fpoly, x)
    Given a set of polynomial coefficients from highest order to x^0,
    compute the value of the polynomail at x.  We assume zero coefficients
    are present in the coefficient list/tuple.

    Example:  f(x) = 4x^3 + 0x^2 + 9x^1 + 3 evaluated at 5
    polyval([4, 0, 9, 3], 5))
    returns 548
    """
    addingValues = []
    exponents = []
    fpoly = fpoly[::-1]
    for i in range(len(fpoly)):
        exponents.append(x**i)
        addingValues.append(exponents[i] * fpoly[i])
    finalValue = sum(addingValues)
    return finalValue

