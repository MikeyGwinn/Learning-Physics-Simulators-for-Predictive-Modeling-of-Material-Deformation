"""Primal-dual active-set solution of the convex AT2 damage subproblem.

Use full-size row/column elimination so CPU and CUDA solve the same bounded
quadratic problem without PETSc SNES VI reduced-matrix backend differences.
"""
import time

import numpy as np


class BoundedDamageProblem:
    def __init__(self, a, L, field, config, timings, linear_iterations):
        import dolfinx.fem as fem
        import dolfinx.fem.petsc as fp
        from petsc4py import PETSc
        from scientific_jit import jit_options
        compilation = jit_options()
        self.J, self.L = fem.form(a, jit_options=compilation), fem.form(L, jit_options=compilation)
        self.u = field
        self.config, self.timings, self.linear_iterations = config, timings, linear_iterations
        self.A = fp.assemble_matrix(self.J)
        self.A.assemble()
        if config["solver"]["backend"] == "gpu":
            self.A.setType("aijcusparse")
        self.x = self.A.createVecRight()
        self.solver = self
        self.reason, self.residual_norm, self.iterations = 0, 0.0, 0
        self.failure_detail = ""

    def setVariableBounds(self, lower, upper):
        self.lower, self.upper = lower, upper

    def getConvergedReason(self):
        return self.reason

    def getFunctionNorm(self):
        return self.residual_norm

    def solve(self):
        import dolfinx.fem.petsc as fp
        from petsc4py import PETSc
        from simulate_fracture import synchronize

        s = self.config["solver"]
        backend = s["backend"]
        synchronize(backend)
        t = time.perf_counter()
        self.A.zeroEntries()
        fp.assemble_matrix(self.A, self.J)
        self.A.assemble()
        rhs_host = fp.assemble_vector(self.L)
        rhs = self.A.createVecLeft()
        rhs.array[:] = rhs_host.array
        rhs_host.destroy()
        diagonal = self.A.getDiagonal()
        diag = diagonal.array.copy()
        diagonal.destroy()
        if np.any(diag <= 0):
            raise RuntimeError("Damage tangent has a nonpositive diagonal")
        synchronize(backend)
        self.timings["jacobian_assembly_s"] += time.perf_counter()-t
        x = self.x
        x.array[:] = self.u.x.array
        lower, upper = self.lower.array.copy(), self.upper.array.copy()
        residual = rhs.duplicate()
        bound_values = x.duplicate()
        self.reason = -5
        self.failure_detail = "Active-set iteration limit"
        previous_active = None
        tolerance = max(s["snes_atol"], 10*s["ksp_rtol"])
        try:
            for iteration in range(1, s["snes_max_it"]+1):
                self.A.mult(x, residual)
                residual.axpy(-1, rhs)
                candidate = x.array-residual.array/diag
                active_lower = candidate <= lower
                active_upper = candidate >= upper
                active = np.flatnonzero(active_lower | active_upper).astype(PETSc.IntType)
                bound_values.set(0)
                bound_values.array[active_lower] = lower[active_lower]
                bound_values.array[active_upper] = upper[active_upper]
                t = time.perf_counter()
                A = self.A.copy()
                b = rhs.copy()
                A.zeroRowsColumns(active, diag=1, x=bound_values, b=b)
                synchronize(backend)
                self.timings["jacobian_assembly_s"] += time.perf_counter()-t
                ksp = PETSc.KSP().create(self.A.comm)
                try:
                    ksp.setOperators(A)
                    if s["linear_solver"] == "direct":
                        ksp.setType("preonly")
                        ksp.getPC().setType("lu")
                        ksp.getPC().setFactorSolverType("mumps")
                    else:
                        ksp.setType("gmres")
                        ksp.setGMRESRestart(100)
                        ksp.getPC().setType(s["preconditioner"])
                        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                        ksp.setTolerances(rtol=min(s["ksp_rtol"], 1e-12), atol=min(s["ksp_atol"], 1e-14), max_it=s["ksp_max_it"])
                    ksp.setInitialGuessNonzero(False)
                    x.set(0)
                    synchronize(backend)
                    t = time.perf_counter()
                    ksp.solve(b, x)
                    synchronize(backend)
                    self.timings["linear_solve_s"] += time.perf_counter()-t
                    self.linear_iterations["d"] += ksp.getIterationNumber()
                    if ksp.getConvergedReason() <= 0:
                        self.reason = -3
                        self.failure_detail = f"KSP reason {ksp.getConvergedReason()}"
                        break
                finally:
                    ksp.destroy(); b.destroy(); A.destroy()
                self.A.mult(x, residual)
                residual.axpy(-1, rhs)
                projected = x.array-np.clip(x.array-residual.array/diag, lower, upper)
                self.residual_norm = float(np.max(np.abs(projected)))
                self.iterations = iteration
                self.failure_detail = f"iteration={iteration}, KKT={self.residual_norm:.3g}, lower_gap={np.min(x.array-lower):.3g}, upper_gap={np.max(x.array-upper):.3g}"
                bounds_ok = np.min(x.array-lower) >= -tolerance and np.max(x.array-upper) <= tolerance
                if bounds_ok and self.residual_norm <= tolerance:
                    self.reason = 2
                    break
                if previous_active is not None and np.array_equal(active, previous_active):
                    # Stable active set but insufficient linear accuracy: report
                    # failure rather than accepting an unconverged KKT point.
                    break
                previous_active = active.copy()
            self.u.x.array[:] = x.array
            self.u.x.scatter_forward()
            return self.u
        finally:
            bound_values.destroy(); residual.destroy(); rhs.destroy()
