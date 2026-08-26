import torch 

def legendre(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n

def legendre_derivative(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    else:
        # The quotient formula is singular at the endpoints even though the
        # polynomial derivative is finite there.  Gauss-Lobatto quadrature
        # includes both endpoints, so use the exact limiting values instead
        # of the old (and mathematically incorrect) zero substitution.
        at_right = x == 1
        at_left = x == -1
        at_endpoint = at_right | at_left
        denominator = torch.where(at_endpoint, torch.ones_like(x), 1 - x**2)
        interior = n * (legendre(n - 1, x) - x * legendre(n, x)) / denominator
        endpoint_magnitude = n * (n + 1) / 2
        right = torch.full_like(x, endpoint_magnitude)
        left = torch.full_like(x, ((-1) ** (n + 1)) * endpoint_magnitude)
        return torch.where(at_right, right, torch.where(at_left, left, interior))


def legendre_bubble(n, x):
    """Legendre bubble ``P_{n+1} - P_{n-1}`` for ``n >= 1``."""

    if n < 1:
        raise ValueError("Legendre bubble index must be at least 1.")
    return legendre(n + 1, x) - legendre(n - 1, x)


def legendre_bubble_derivative(n, x):
    """Derivative of :func:`legendre_bubble` on the reference cell."""

    if n < 1:
        raise ValueError("Legendre bubble index must be at least 1.")
    return legendre_derivative(n + 1, x) - legendre_derivative(n - 1, x)

    
def u(k, n, x):
    if k == 0:
        return legendre(n, x)
    elif k == 1:
        return legendre_derivative(n, x)
    else:
        raise ValueError("k must be 0 or 1")

def v2d(k, n1, n2, x, y):
    if k == 0:
        return legendre_bubble(n1, x) * legendre_bubble(n2, y)
    
    elif k == 1:
        return torch.cat([
            legendre_bubble_derivative(n1, x) * legendre_bubble(n2, y),
            legendre_bubble(n1, x) * legendre_bubble_derivative(n2, y),
        ], dim=1)

def v3d(k, n1, n2, n3, x, y, z):
    if k == 0:
        return legendre_bubble(n1, x) * legendre_bubble(n2, y) * legendre_bubble(n3, z)
    
    # elif k == 1:
    #     return torch.cat([(legendre_derivative(n1 + 1, x) - legendre_derivative(n1 - 1, x)) * \
    #         (legendre(n2 + 1, y) - legendre(n2 - 1, y)), (legendre(n1 + 1, x) - legendre(n1 - 1, x)) \
    #             * (legendre_derivative(n2 + 1, y) - legendre_derivative(n2 - 1, y))], dim=1)

class Test_Func:
    
    def init(self, test_func_num):
        self.test_func_num = test_func_num
        
    def test_func(self, k, x, y, z=None):
        ret = []
        
        if self.test_func_num==0:
            return torch.ones_like(x)
        
        else:
            if z == None:
                for n1 in range(1, self.test_func_num + 1):
                    for n2 in range(1, self.test_func_num + 1):
                        ret.append(v2d(k, n1, n2, x, y))
                return torch.stack(ret, dim=0)
            else:
                for n1 in range(1, self.test_func_num + 1):
                    for n2 in range(1, self.test_func_num + 1):
                        for n3 in range(1, self.test_func_num + 1):
                            ret.append(v3d(k, n1, n2, n3, x, y, z))
                return torch.stack(ret, dim=0)
            # return torch.cat(ret, dim=1).t()
        # return torch.sum(torch.tensor([[[v(k, n1, n2, x, y)] for n1 in range(self.test_func_num)] for n2 in range(self.test_func_num)]))
    
test_func = Test_Func()
