import numpy as np

"""## C01"""

class ProblemaC01:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, 0.0)
        self.upper_bounds = np.full(self.D, 10.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        cos2        = np.cos(self.z) ** 2
        cos4        = np.cos(self.z) ** 4
        numerador   = np.abs(np.sum(cos4) - 2 * np.prod(cos2))
        denominador = np.sqrt(np.sum(np.arange(1, self.D + 1) * self.z ** 2))
        if denominador < self.tolerance:
            return 0.0
        return - (numerador / denominador)

    def g1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return 0.75 - np.prod(z)

    def g2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z) - 7.5 * self.D

    def sumar_violation(self):
        viol_g1 = max(0.0, self.g1())
        viol_g2 = max(0.0, self.g2())
        return viol_g1 + viol_g2

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Vector de violaciones: usa g1(x) y g2(x)
        deltaC = np.array([
            max(0.0, self.g1(x)),
            max(0.0, self.g2(x))
        ])
        # Construir Jacobiano usando g1, g2 y step de diferencia finita = tolerance
        funcs = [lambda v: self.g1(v),
                 lambda v: self.g2(v)]
        J = np.vstack([
            self._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

    @staticmethod
    def _num_grad(func, x, eps):
        grad = np.zeros_like(x, dtype=float)
        for i in range(x.size):
            x_fwd = x.copy(); x_bwd = x.copy()
            x_fwd[i] += eps; x_bwd[i] -= eps
            grad[i] = (func(x_fwd) - func(x_bwd)) / (2 * eps)
        return grad

"""## C02"""

class ProblemaC02:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -5.12)
        self.upper_bounds = np.full(self.D,  5.12)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        self.y = self.z - 0.5
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        return np.max(self.z)

    def g1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        restr = z**2 - 10 * np.cos(2 * np.pi * z) + 10
        return 10 - (1 / self.D) * np.sum(restr)

    def g2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        restr = z**2 - 10 * np.cos(2 * np.pi * z) + 10
        return (1 / self.D) * np.sum(restr) - 15

    def h1(self, x=None):
        if x is None:
            y = self.y
        else:
            y = (np.asarray(x, dtype=float) - self.offset) - 0.5
        restr = y**2 - 10 * np.cos(2 * np.pi * y) + 10
        return (1 / self.D) * np.sum(restr) - 20

    def sumar_violation(self):
        v1 = max(0.0, self.g1())
        v2 = max(0.0, self.g2())
        v3 = max(0.0, abs(self.h1()) - self.tolerance)
        return v1 + v2 + v3

    def viol_and_jac(self, individuo):

        x = np.asarray(individuo, dtype=float)
        # Recalcular internos
        self.z = x - self.offset
        self.y = self.z - 0.5
        # deltaC con tus métodos
        deltaC = np.array([
            max(0.0, self.g1(x)),
            max(0.0, self.g2(x)),
            max(0.0, abs(self.h1(x)) - self.tolerance)
        ])
        # Jacobiana con _num_grad de C01
        funcs = [lambda v: self.g1(v),
                 lambda v: self.g2(v),
                 lambda v: self.h1(v)]
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C03"""

class ProblemaC03:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -1000.0)
        self.upper_bounds = np.full(self.D,  1000.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        term1 = 100 * (self.z[:-1]**2 - self.z[1:])**2
        term2 = (self.z[:-1] - 1)**2
        return np.sum(term1 + term2)

    def h1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        violation = np.abs(np.sum((z[:-1] - z[1:])**2))
        return violation

    def sumar_violation(self):
        viol_h = max(0.0, abs(self.h1()) - self.tolerance)
        return viol_h

    def viol_and_jac(self, individuo):
        """
        Calcula deltaC y Jacobiano J(x) para la restricción de C03.
        """
        x = np.asarray(individuo, dtype=float)
        self.z = x - self.offset
        # deltaC
        deltaC = np.array([max(0.0, abs(self.h1(x)) - self.tolerance)])
        # Jacobiana con _num_grad de C01 usando h1
        func = lambda v: self.h1(v)
        J    = np.vstack([ProblemaC01._num_grad(func, x, self.tolerance)])
        return deltaC, J

"""## C04"""

class ProblemaC04:
    def __init__(self, offset):
        # Tolerancia para igualdad h
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -50.0)
        self.upper_bounds = np.full(self.D,  50.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.suma_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        return np.max(self.z)

    def h1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z * np.cos(np.sqrt(np.abs(z)))) / self.D

    def h2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        upper = int(self.D / 2) - 1
        diffs = z[:upper] - z[1:upper+1]
        return np.sum(diffs**2)

    def h3(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        if self.D < 4:
            return 0.0
        start = int(self.D / 2)
        diffs = z[start:-1]**2 - z[start+1:]
        return np.sum(diffs**2)

    def h4(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z)

    def suma_violation(self):
        v1 = max(0.0, abs(self.h1()) - self.tolerance)
        v2 = max(0.0, abs(self.h2()) - self.tolerance)
        v3 = max(0.0, abs(self.h3()) - self.tolerance)
        v4 = max(0.0, abs(self.h4()) - self.tolerance)
        return v1 + v2 + v3 + v4

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z
        self.z = x - self.offset
        # Vector de violaciones
        deltaC = np.array([
            max(0.0, abs(self.h1(x)) - self.tolerance),
            max(0.0, abs(self.h2(x)) - self.tolerance),
            max(0.0, abs(self.h3(x)) - self.tolerance),
            max(0.0, abs(self.h4(x)) - self.tolerance)
        ])
        # Funciones restricciones para gradiente
        funcs = [lambda v: self.h1(v),
                 lambda v: self.h2(v),
                 lambda v: self.h3(v),
                 lambda v: self.h4(v)]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C05"""

class ProblemaC05:
    def __init__(self, offset):
        # Tolerancia para las igualdades
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -600.0)
        self.upper_bounds = np.full(self.D,  600.0)

    def get_limites(self):
        """Devuelve los límites inferiores y superiores."""
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        # f(x) = max(z)
        return np.max(self.z)

    def h1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(-z * np.sin(np.sqrt(np.abs(z)))) / self.D

    def h2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(-z * np.cos(0.5 * np.sqrt(np.abs(z)))) / self.D

    def sumar_violation(self):
        v1 = max(0.0, abs(self.h1()) - self.tolerance)
        v2 = max(0.0, abs(self.h2()) - self.tolerance)
        return v1 + v2

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z
        self.z = x - self.offset
        # Vector de violaciones
        deltaC = np.array([
            max(0.0, abs(self.h1(x)) - self.tolerance),
            max(0.0, abs(self.h2(x)) - self.tolerance)
        ])
        # Funciones restricciones para gradiente
        funcs = [lambda v: self.h1(v),
                 lambda v: self.h2(v)]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C06"""

class ProblemaC06:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -600)
        self.upper_bounds = np.full(self.D,  600)
        self.m            = self.generar_m(42)
    
    def get_limites(self):
        return self.lower_bounds, self.upper_bounds
    
    def generar_m(self, seed=None):
        rng = np.random.RandomState(seed)
        A = rng.randn(self.D, self.D)
        Q, _ = np.linalg.qr(A)
        return Q
    
    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        self.y = (self.x + 483.6106156535 - self.offset) @ self.m - 483.6106156535
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones
    
    def aptitud(self):
        return np.max(self.z)
    
    def h1(self, x=None):
        if x is None:
            y = self.y
        else:
            z = np.asarray(x, dtype=float) - self.offset
            y = (x + 483.6106156535 - self.offset) @ self.m - 483.6106156535
        restr = -y * np.sin(np.sqrt(np.abs(y)))
        return np.sum(restr) / self.D
    
    def h2(self, x=None):
        if x is None:
            y = self.y
        else:
            y = (np.asarray(x, dtype=float) + 483.6106156535 - self.offset) @ self.m - 483.6106156535
        restr = -y * np.cos(0.5 * np.sqrt(np.abs(y)))
        return np.sum(restr) / self.D
    
    def sumar_violation(self):
        v1 = max(0.0, abs(self.h1()) - self.tolerance)
        v2 = max(0.0, abs(self.h2()) - self.tolerance)
        return v1 + v2
    
    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z e y
        self.z = x - self.offset
        self.y = (x + 483.6106156535 - self.offset) @ self.m - 483.6106156535
        # Vector de violaciones
        deltaC = np.array([
            max(0.0, abs(self.h1(x)) - self.tolerance),
            max(0.0, abs(self.h2(x)) - self.tolerance)
        ])
        # Funciones restricciones para gradiente
        funcs = [lambda v: self.h1(v),
                 lambda v: self.h2(v)]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C07"""

class ProblemaC07:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -140.0)
        self.upper_bounds = np.full(self.D,  140.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x + 1 - self.offset
        self.y = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        term1 = 100 * (self.z[:-1]**2 - self.z[1:])**2
        term2 = (self.z[:-1] - 1)**2
        return np.sum(term1 + term2)

    def g1(self, x=None):
        if x is None:
            y = self.y
        else:
            y = np.asarray(x, dtype=float) - self.offset
        term1 = np.exp(-0.1 * np.sqrt(np.sum(y**2) / self.D))
        term2 = 3 * np.exp(np.sum(np.cos(0.1 * y)) / self.D)
        return 0.5 - term1 - term2 + np.exp(1)

    def sumar_violation(self):
        return max(0.0, self.g1())

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        self.x = x
        self.z = x + 1 - self.offset
        self.y = x - self.offset
        # Vector de violaciones
        deltaC = np.array([max(0.0, self.g1())])
        # Función para gradiente usando g1(x)
        func = lambda v: ProblemaC07.g1(self, v)
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([ProblemaC01._num_grad(func, x, self.tolerance)])
        return deltaC, J

"""## C08"""

class ProblemaC08:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -140.0)
        self.upper_bounds = np.full(self.D,  140.0)
        self.m            = self.generar_m(42)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def generar_m(self, seed=None):
        rng = np.random.RandomState(seed)
        A = rng.randn(self.D, self.D)
        Q, _ = np.linalg.qr(A)
        return Q

    def evaluate(self, individuo):
        self.x               = np.asarray(individuo, dtype=float)
        self.z               = self.x + 1 - self.offset
        self.y               = (self.x - self.offset) @ self.m
        fitness              = self.aptitud()
        suma_violaciones     = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        term1 = 100 * (self.z[:-1]**2 - self.z[1:])**2
        term2 = (self.z[:-1] - 1)**2
        return np.sum(term1 + term2)

    def g1(self, x=None):
        if x is None:
            y = self.y
        else:
            z = np.asarray(x, dtype=float) - self.offset
            y = z @ self.m
        term1 = 0.5 - np.exp(-0.1 * np.sqrt((1 / self.D) * np.sum(y**2)))
        term2 = 3 - np.exp((1 / self.D) * np.sum(np.cos(0.1 * y)))
        return term1 - term2 + np.exp(1)

    def sumar_violation(self):
        v1 = max(0.0, self.g1())
        return v1

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        self.z = x + 1 - self.offset
        self.y = (x - self.offset) @ self.m
        deltaC = np.array([max(0.0, self.g1(x))])
        func = lambda v: self.g1(v)
        J    = np.vstack([ProblemaC01._num_grad(func, x, self.tolerance)])
        return deltaC, J

"""## C09"""

class ProblemaC09:
    def __init__(self, offset):
        # Tolerancia para la igualdad
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -500.0)
        self.upper_bounds = np.full(self.D,  500.0)

    def get_limites(self):
        """Devuelve los límites inferiores y superiores."""
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x + 1.0 - self.offset
        self.y = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        # f(x) = Σ_{i=1 to D-1} [ 100 ( z_i^2 – z_{i+1} )^2 + ( z_i – 1 )^2 ]
        diffs = self.z[:-1]**2 - self.z[1:]
        return np.sum(100.0 * diffs**2 + (self.z[:-1] - 1.0)**2)

    def h1(self, x=None):
        if x is None:
            y = self.y
        else:
            y = np.asarray(x, dtype=float) - self.offset
        # h(x) = Σ y_i sin(√|y_i|) = 0
        return np.sum(y * np.sin(np.sqrt(np.abs(y))))

    def sumar_violation(self):
        return max(0.0, abs(self.h1()) - self.tolerance)

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular intermedios
        self.z = x + 1.0 - self.offset
        self.y = x - self.offset
        # deltaC: violación de h1
        deltaC = np.array([max(0.0, abs(self.h1()) - self.tolerance)])
        # Wrapper para gradiente usando h1 existente
        def _func(v):
            self.y = np.asarray(v, dtype=float) - self.offset
            return self.h1()
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([ProblemaC01._num_grad(_func, x, self.tolerance)])
        return deltaC, J

"""## C10"""

class ProblemaC10:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -500.0)
        self.upper_bounds = np.full(self.D,  500.0)
        self.m            = self.generar_m(seed=42)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def generar_m(self, seed=None):
        rng = np.random.RandomState(seed)
        A = rng.randn(self.D, self.D)
        Q, _ = np.linalg.qr(A)
        return Q

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        if self.x.shape != self.offset.shape:
            raise ValueError(
                f"Dimensionalidad de individuo ({len(self.x)}) "
                f"no coincide con offset ({len(self.offset)})"
            )
        self.z = self.x + 1 - self.offset
        self.y = (self.x - self.offset) @ self.m
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        term1 = 100 * (self.z[:-1]**2 - self.z[1:])**2
        term2 = (self.z[:-1] - 1)**2
        return np.sum(term1 + term2)

    def h1(self, x=None):
        if x is None:
            y = self.y
        else:
            y = (np.asarray(x, dtype=float) - self.offset) @ self.m
        return np.sum(y * np.sin(np.sqrt(np.abs(y))))

    def sumar_violation(self):
        h1_val = self.h1()
        return max(0.0, abs(h1_val) - self.tolerance)

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z e y
        self.z = x + 1 - self.offset
        self.y = (x - self.offset) @ self.m
        # Construir deltaC usando h1
        deltaC = np.array([max(0.0, abs(self.h1(x)) - self.tolerance)])
        # Función wrapper para gradiente utilizando h1
        func = lambda v: ProblemaC10.h1(self, v)
        # Jacobiana con _num_grad de ProblemaC01
        J = np.vstack([ProblemaC01._num_grad(func, x, self.tolerance)])
        return deltaC, J

"""## C11"""

class ProblemaC11:
    def __init__(self, offset):
        """Ctor fixes offset, tolerance, matrix Q"""
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -100.0)
        self.upper_bounds = np.full(self.D,  100.0)
        self.m            = self.generar_m(seed=42)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def generar_m(self, seed=None):
        rng = np.random.RandomState(seed)
        A = rng.randn(self.D, self.D)
        Q, _ = np.linalg.qr(A)
        return Q

    def evaluate(self, individuo):
        # asegura la forma correcta y fija x, z, y
        self.x = np.asarray(individuo, dtype=float)
        if self.x.shape != self.offset.shape:
            raise ValueError(
                f"Dimensionalidad de individuo ({len(self.x)}) "
                f"no coincide con offset ({len(self.offset)})"
            )
        self.z = (self.x - self.offset) @ self.m
        self.y = self.x + 1 - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        # función objetivo de C11
        return np.mean(-self.z * np.cos(2 * np.sqrt(np.abs(self.z))))

    def h1(self, x=None):
        # restricción igualitaria relajada como |h1|<=tol
        if x is None:
            y = self.y
        else:
            y = np.asarray(x, dtype=float) + 1 - self.offset
        term1 = 100 * (y[:-1]**2 - y[1:])**2
        term2 = (y[:-1] - 1)**2
        return np.sum(term1 + term2)

    def sumar_violation(self):
        h1_val = self.h1()
        return max(0.0, abs(h1_val) - self.tolerance)

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # recalcula internals
        self.z = (x - self.offset) @ self.m
        self.y = x + 1 - self.offset
        # deltaC: violación de h1
        deltaC = np.array([max(0.0, abs(self.h1(x)) - self.tolerance)])
        # wrapper para gradiente
        func   = lambda v: ProblemaC11.h1(self, v)
        # usa _num_grad de C01
        J      = np.vstack([ProblemaC01._num_grad(func, x, self.tolerance)])
        return deltaC, J

"""## C12"""

class ProblemaC12:
    def __init__(self, offset):
        # Tolerancia para igualdad h
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -1000.0)
        self.upper_bounds = np.full(self.D,  1000.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        # Valida offset inicializado
        if self.offset is None or self.D is None:
            raise ValueError("Antes de evaluar, debes llamar a 'limites(offset)' para fijar el offset.")
        x = np.asarray(individuo, dtype=float)
        if x.shape != self.offset.shape:
            raise ValueError(
                f"Dimensionalidad de individuo ({len(x)}) "
                f"no coincide con offset ({len(self.offset)})"
            )
        z = x - self.offset
        fitness          = self.aptitud(z)
        suma_violaciones = self.sumar_violation(z)
        return fitness, suma_violaciones

    def aptitud(self, z):
        # Función objetivo C12
        inner = z * np.sin(np.sqrt(np.abs(z)))
        return np.sum(inner)

    def h(self, z):
        # Igualdad relajada
        diffs = z[:-1]**2 - z[1:]
        return np.sum(diffs**2)

    def g(self, z):
        # Desigualdad
        inner = z - 100 * np.cos(0.1 * z) + 10
        return np.sum(inner)

    def sumar_violation(self, z):
        # Violaciones de h y g
        viol_h = max(0.0, abs(self.h(z)) - self.tolerance)
        viol_g = max(0.0, self.g(z))
        return viol_h + viol_g

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Reconstruir z
        z = x - self.offset
        # Vector de violaciones: [h_violation, g_violation]
        deltaC = np.array([
            max(0.0, abs(self.h(z)) - self.tolerance),
            max(0.0, self.g(z))
        ])
        # Funciones wrapper para gradiente en x-space
        f_h = lambda v: self.h(np.asarray(v, dtype=float) - self.offset)
        f_g = lambda v: self.g(np.asarray(v, dtype=float) - self.offset)
        # Jacobiana usando numerial grad
        J = np.vstack([
            ProblemaC01._num_grad(f_h, x, self.tolerance),
            ProblemaC01._num_grad(f_g, x, self.tolerance)
        ])
        return deltaC, J

"""## C13"""

class ProblemaC13:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -500.0)
        self.upper_bounds = np.full(self.D,  500.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        return np.mean(-self.z * np.sin(np.sqrt(np.abs(self.z))))

    def g1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return -50 + (1 / (100 * self.D)) * np.sum(z**2)

    def g2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return (50 / self.D) * np.sum(np.sin(np.abs(z) / 50)) - np.pi

    def g3(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        sum_term  = np.sum(z**2) / 4000
        prod_term = np.prod(np.cos(z / np.sqrt(np.arange(1, self.D + 1))))
        return 75 - 50 * (sum_term - prod_term + 1)

    def sumar_violation(self):
        viol_g1 = max(0.0, self.g1())
        viol_g2 = max(0.0, self.g2())
        viol_g3 = max(0.0, self.g3())
        return viol_g1 + viol_g2 + viol_g3

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z
        self.z = x - self.offset
        # Vector de violaciones: g1, g2, g3
        deltaC = np.array([
            max(0.0, self.g1(x)),
            max(0.0, self.g2(x)),
            max(0.0, self.g3(x))
        ])
        # Funciones restricciones para gradiente
        funcs = [lambda v: self.g1(v),
                 lambda v: self.g2(v),
                 lambda v: self.g3(v)]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C14"""

class ProblemaC14:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -1000.0)
        self.upper_bounds = np.full(self.D,  1000.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        x = np.asarray(individuo, dtype=float)
        z = x + 1 - self.offset
        y = x - self.offset
        fitness          = self.aptitud(z)
        suma_violaciones = self.sumar_violation(y)
        return fitness, suma_violaciones

    def aptitud(self, z):
        return np.sum(100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2)

    def g1(self, y):
        return np.sum(-y * np.cos(np.sqrt(np.abs(y)))) - self.D

    def g2(self, y):
        return np.sum(y * np.cos(np.sqrt(np.abs(y)))) - self.D

    def g3(self, y):
        return np.sum(y * np.sin(np.sqrt(np.abs(y)))) - 10 * self.D

    def sumar_violation(self, y):
        v1 = max(0.0, self.g1(y))
        v2 = max(0.0, self.g2(y))
        v3 = max(0.0, self.g3(y))
        return v1 + v2 + v3

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular intermedios
        z = x + 1 - self.offset
        y = x - self.offset
        # Vector de violaciones: g1, g2, g3
        deltaC = np.array([
            max(0.0, self.g1(y)),
            max(0.0, self.g2(y)),
            max(0.0, self.g3(y))
        ])
        # Wrappers para gradiente
        f1 = lambda v: self.g1(np.asarray(v, dtype=float) - self.offset)
        f2 = lambda v: self.g2(np.asarray(v, dtype=float) - self.offset)
        f3 = lambda v: self.g3(np.asarray(v, dtype=float) - self.offset)
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f1, x, self.tolerance),
            ProblemaC01._num_grad(f2, x, self.tolerance),
            ProblemaC01._num_grad(f3, x, self.tolerance)
        ])
        return deltaC, J

"""## C15"""

class ProblemaC15:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -1000.0)
        self.upper_bounds = np.full(self.D,  1000.0)
        self.m            = self.generar_m(42)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def generar_m(self, seed=None):
        rng = np.random.RandomState(seed)
        A = rng.randn(self.D, self.D)
        Q, _ = np.linalg.qr(A)
        return Q

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x + 1 - self.offset
        self.y = (self.x - self.offset) @ self.m
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        term1 = 100.0 * (self.z[:-1]**2 - self.z[1:])**2
        term2 = (self.z[:-1] - 1)**2
        return np.sum(term1 + term2)

    def g1(self, x=None):
        if x is None:
            y = self.y
        else:
            y = (np.asarray(x, dtype=float) - self.offset) @ self.m
        return np.sum(-y * np.cos(np.sqrt(np.abs(y)))) - self.D

    def g2(self, x=None):
        if x is None:
            y = self.y
        else:
            y = (np.asarray(x, dtype=float) - self.offset) @ self.m
        return np.sum(y * np.cos(np.sqrt(np.abs(y)))) - self.D

    def g3(self, x=None):
        if x is None:
            y = self.y
        else:
            y = (np.asarray(x, dtype=float) - self.offset) @ self.m
        return np.sum(y * np.sin(np.sqrt(np.abs(y)))) - 10 * self.D

    def sumar_violation(self):
        v1 = max(0.0, self.g1())
        v2 = max(0.0, self.g2())
        v3 = max(0.0, self.g3())
        return v1 + v2 + v3

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z e y
        self.z = x + 1 - self.offset
        self.y = (x - self.offset) @ self.m
        # Vector de violaciones usando g1,g2,g3
        deltaC = np.array([
            max(0.0, self.g1(x)),
            max(0.0, self.g2(x)),
            max(0.0, self.g3(x))
        ])
        # Funciones wrapper para gradiente
        funcs = [lambda v: self.g1(v),
                 lambda v: self.g2(v),
                 lambda v: self.g3(v)]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C16"""

class ProblemaC16:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -10.0)
        self.upper_bounds = np.full(self.D,  10.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.suma_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        sum_term  = np.sum(self.z**2) / 4000.0
        prod_term = np.prod(np.cos(self.z / np.sqrt(np.arange(1, self.D + 1))))
        return sum_term - prod_term + 1

    def g1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z**2 - 100 * np.cos(np.pi * z) + 10)

    def g2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.prod(z)

    def h1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z * np.sin(np.sqrt(np.abs(z))))

    def h2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(-z * np.sin(np.sqrt(np.abs(z))))

    def suma_violation(self):
        v_g1 = max(0.0, self.g1())
        v_g2 = max(0.0, self.g2())
        v_h1 = max(0.0, abs(self.h1()) - self.tolerance)
        v_h2 = max(0.0, abs(self.h2()) - self.tolerance)
        return v_g1 + v_g2 + v_h1 + v_h2

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z
        self.z = x - self.offset
        # Vector de violaciones: g1, g2, h1, h2
        deltaC = np.array([
            max(0.0, self.g1(x)),
            max(0.0, self.g2(x)),
            max(0.0, abs(self.h1(x)) - self.tolerance),
            max(0.0, abs(self.h2(x)) - self.tolerance)
        ])
        # Funciones restricciones para gradiente
        funcs = [lambda v: self.g1(v),
                 lambda v: self.g2(v),
                 lambda v: self.h1(v),
                 lambda v: self.h2(v)]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C17"""

class ProblemaC17:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -10.0)
        self.upper_bounds = np.full(self.D,  10.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        self.x = np.asarray(individuo, dtype=float)
        self.z = self.x - self.offset
        fitness          = self.aptitud()
        suma_violaciones = self.sumar_violation()
        return fitness, suma_violaciones

    def aptitud(self):
        return np.sum((self.z[:-1] - self.z[1:]) ** 2)

    def g1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.prod(z)

    def g2(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z)

    def h1(self, x=None):
        if x is None:
            z = self.z
        else:
            z = np.asarray(x, dtype=float) - self.offset
        return np.sum(z * np.sin(4 * np.sqrt(np.abs(z))))

    def sumar_violation(self):
        v1 = max(0.0, self.g1())
        v2 = max(0.0, self.g2())
        v3 = max(0.0, abs(self.h1()) - self.tolerance)
        return v1 + v2 + v3

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Recalcular z
        self.z = x - self.offset
        # deltaC: g1, g2, h1 violaciones
        deltaC = np.array([
            max(0.0, self.g1(x)),
            max(0.0, self.g2(x)),
            max(0.0, abs(self.h1(x)) - self.tolerance)
        ])
        # Funciones de restricción para gradiente
        funcs = [
            lambda v: self.g1(v),
            lambda v: self.g2(v),
            lambda v: self.h1(v)
        ]
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f, x, self.tolerance) for f in funcs
        ])
        return deltaC, J

"""## C18"""

class ProblemaC18:
    def __init__(self, offset):
        self.tolerance    = 1e-4
        self.offset       = np.asarray(offset, dtype=float)
        self.D            = len(self.offset)
        self.lower_bounds = np.full(self.D, -50.0)
        self.upper_bounds = np.full(self.D,  50.0)

    def get_limites(self):
        return self.lower_bounds, self.upper_bounds

    def evaluate(self, individuo):
        x = np.asarray(individuo, dtype=float)
        z = x - self.offset
        fitness          = self.aptitud(z)
        suma_violaciones = self.sumar_violation(z)
        return fitness, suma_violaciones

    def aptitud(self, z):
        diffs = z[:-1] - z[1:]
        return np.sum(diffs**2)

    def g(self, z):
        return (1 / self.D) * np.sum(-z * np.sin(np.sqrt(np.abs(z))))

    def h(self, z):
        return (1 / self.D) * np.sum(z * np.sin(np.sqrt(np.abs(z))))

    def sumar_violation(self, z):
        h_val = self.h(z)
        g_val = self.g(z)
        viol_h = max(0.0, abs(h_val) - self.tolerance)
        viol_g = max(0.0, g_val)
        return viol_h + viol_g

    def viol_and_jac(self, individuo):
        x = np.asarray(individuo, dtype=float)
        # Reconstruir z
        z = x - self.offset
        # Vector de violaciones: [h_violation, g_violation]
        deltaC = np.array([
            max(0.0, abs(self.h(z)) - self.tolerance),
            max(0.0, self.g(z))
        ])
        # Wrappers para gradiente en espacio x
        f_h = lambda v: self.h(np.asarray(v, dtype=float) - self.offset)
        f_g = lambda v: self.g(np.asarray(v, dtype=float) - self.offset)
        # Jacobiana usando _num_grad de ProblemaC01
        J = np.vstack([
            ProblemaC01._num_grad(f_h, x, self.tolerance),
            ProblemaC01._num_grad(f_g, x, self.tolerance)
        ])
        return deltaC, J
