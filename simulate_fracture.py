"""Quasi-static AT2 fracture of an explicitly porous plate in horizontal tension.

Single rank, P1 triangles, Amor volumetric/deviatoric split, accepted-load
DG0 tensile history and nodal damage bounds. Units: mm, N, MPa, N/mm.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from scientific_common import DEFAULT_CONFIG, load_config, validate_config, write_json
from scientific_jit import jit_options


def synchronize(backend):
    if backend == "gpu":
        from petsc4py import PETSc
        PETSc.DeviceContext.getCurrent().synchronize()


def solver_options(c):
    s = c["solver"]
    options = {"snes_type": "newtonls",
               "snes_linesearch_type": "bt", "snes_rtol": s["snes_rtol"],
               "snes_atol": s["snes_atol"], "snes_max_it": s["snes_max_it"],
               "ksp_type": "gmres", "pc_type": s["preconditioner"],
               "ksp_rtol": s["ksp_rtol"], "ksp_atol": s["ksp_atol"],
               "ksp_max_it": s["ksp_max_it"], "ksp_gmres_restart": 100,
               "ksp_norm_type": "unpreconditioned", "ksp_error_if_not_converged": False}
    if s["linear_solver"] == "direct":
        options.update(ksp_type="preonly", pc_type="lu", pc_factor_mat_solver_type="mumps")
        options.pop("ksp_gmres_restart")
        for key in ("ksp_rtol", "ksp_atol", "ksp_max_it", "ksp_norm_type"):
            options.pop(key)
    return options


def run_simulation(config, output, mesh_input=None):
    if mesh_input is None:
        raise ValueError("SEM fracture simulation requires the generated image mesh")
    import dolfinx
    import dolfinx.fem.petsc as fp
    import dolfinx.plot
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    from bounded_damage import BoundedDamageProblem

    validate_config(config)
    compilation = jit_options()
    print(f"FEM compilation cache: {compilation['cache_dir']}", flush=True)
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Use one MPI rank; multi-rank output/replay is not supported")
    backend = config["solver"]["backend"]
    if backend == "gpu" and not PETSc.Sys.hasExternalPackage("cuda"):
        raise RuntimeError("GPU requested but PETSc has no CUDA support")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"status": "running"})
    start_setup = time.perf_counter()
    timings = {"residual_assembly_s": 0.0, "jacobian_assembly_s": 0.0,
               "linear_solve_s": 0.0, "displacement_s": 0.0, "damage_s": 0.0,
               "history_s": 0.0, "diagnostics_s": 0.0, "field_output_s": 0.0}
    try:
        from dolfinx.io import gmsh as dx_gmsh
        mesh_folder = Path(mesh_input).resolve()
        geo = json.loads((mesh_folder / "geometry.json").read_text())
        if not geo.get("ready_for_fracture"):
            raise ValueError("Image mesh must have one connected solid component spanning both grips")
        if not np.isclose(config["geometry"]["width"], geo["width"], rtol=1e-10, atol=1e-15) or not np.isclose(config["geometry"]["height"], geo["height"], rtol=1e-10, atol=1e-15):
            raise ValueError("Configuration dimensions differ from the calibrated image mesh")
        data = dx_gmsh.read_from_msh(str(mesh_folder / "specimen.msh"), comm, rank=0, gdim=2)
        mesh, facets = data.mesh, data.facet_tags
        import shutil
        shutil.copyfile(mesh_folder / "specimen.msh", output / "specimen.msh")
        write_json(output / "geometry.json", geo)
        m, s, load = config["material"], config["solver"], config["loading"]
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
        D = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
        u = dolfinx.fem.Function(V, name="displacement")
        d = dolfinx.fem.Function(D, name="damage")
        H = dolfinx.fem.Function(Q, name="tensile_history")
        current_energy = dolfinx.fem.Function(Q)
        vu, vd = ufl.TestFunction(V), ufl.TestFunction(D)
        
        # Only x is constrained along the grips; vertical motion free.
        if facets is None or not len(facets.find(1)) or not len(facets.find(2)):
            raise ValueError("Mesh is missing left/right grip facet tags")
        left = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
        right = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
        dl = dolfinx.fem.locate_dofs_topological(V.sub(0), 1, facets.find(1))
        dr = dolfinx.fem.locate_dofs_topological(V.sub(0), 1, facets.find(2))

        # Image crops may have a pore at the bottom-left corner. Anchor one
        # existing left-grip vertex instead of inventing a material corner.
        mesh.topology.create_connectivity(1, 0)
        connectivity = mesh.topology.connectivity(1, 0)
        candidates = np.unique(np.concatenate([connectivity.links(int(f)) for f in facets.find(1)]))
        # Imported Gmsh meshes may not have the reverse vertex-to-cell graph.
        # DOLFINx compute_midpoints for vertices requires both directions.
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        mesh.topology.create_connectivity(0, mesh.topology.dim)
        coordinates = dolfinx.mesh.compute_midpoints(mesh, 0, candidates)
        vertices = candidates[[int(np.argmin(coordinates[:, 1]))]]
        pin = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, vertices)
        if len(pin) != 1 or not len(dl) or not len(dr):
            raise RuntimeError("Missing grips or rigid-translation gauge vertex")
        bcs = [dolfinx.fem.dirichletbc(left, dl, V.sub(0)),
               dolfinx.fem.dirichletbc(right, dr, V.sub(0)),
               dolfinx.fem.dirichletbc(PETSc.ScalarType(0), pin, V.sub(1))]
        eps = ufl.variable(ufl.sym(ufl.grad(u)))
        mu = m["E"] / (2*(1+m["nu"]))
        lam = m["E"]*m["nu"]/((1+m["nu"])*(1-2*m["nu"]))
        if m["kinematics"] == "plane_stress":
            # Reduced 2D elastic energy: split its trace with the 2D bulk modulus.
            lam = m["E"]*m["nu"]/(1-m["nu"]**2)
            bulk = lam + mu
            dev_energy = mu*(ufl.inner(eps, eps)-ufl.tr(eps)**2/2)
        else:
            # Plane strain uses the full 3D deviator with epsilon_zz=0.
            bulk = lam + 2*mu/3
            dev_energy = mu*(ufl.inner(eps, eps)-ufl.tr(eps)**2/3)
        trace = ufl.tr(eps)
        positive = ufl.conditional(ufl.gt(trace, 0), trace, 0)
        negative = ufl.conditional(ufl.lt(trace, 0), trace, 0)
        psi_plus = bulk/2 * positive**2 + dev_energy
        psi_minus = bulk/2 * negative**2
        degradation = (1-m["residual_stiffness"])*(1-d)**2+m["residual_stiffness"]
        elastic_density = degradation*psi_plus+psi_minus
        stress = ufl.diff(elastic_density, eps)
        dx = ufl.Measure("dx", domain=mesh)
        Fu = ufl.inner(stress, ufl.sym(ufl.grad(vu)))*dx
        problems = []
        linear_iterations = {"u": 0, "d": 0}
        for label, residual, field, constraints in [("u", Fu, u, bcs)]:
            problem = fp.NonlinearProblem(residual, field, bcs=constraints,
                                          jit_options=compilation,
                                          petsc_options_prefix=f"porous_{label}_",
                                          petsc_options=solver_options(config))
            if backend == "gpu":
                fp.assemble_matrix(problem.A, problem.J, bcs=constraints)
                problem.A.assemble()
                problem.A.setType("aijcusparse")
                problem.b.setType("cuda")
                problem.x.setType("cuda")
            # Include host assembly, lifting and backend synchronization in assembly timers.
            def residual_callback(snes, x, b, p=problem, f=field, bc=constraints):
                synchronize(backend)
                t = time.perf_counter()
                fp.assemble_residual(snes, x, b, f, p.F, p.J, bc)
                synchronize(backend)
                timings["residual_assembly_s"] += time.perf_counter()-t

            def jacobian_callback(snes, x, A, P, p=problem, f=field, bc=constraints):
                synchronize(backend)
                t = time.perf_counter()
                fp.assemble_jacobian(snes, x, A, P, f, p.J, p.preconditioner, bc)
                synchronize(backend)
                timings["jacobian_assembly_s"] += time.perf_counter()-t

            problem.solver.setFunction(residual_callback, problem.b)
            problem.solver.setJacobian(jacobian_callback, problem.A)
            clock = [0.0]

            def before(ksp, b, x, clock=clock):
                synchronize(backend)
                clock[0] = time.perf_counter()

            def after(ksp, b, x, clock=clock, label=label):
                synchronize(backend)
                timings["linear_solve_s"] += time.perf_counter()-clock[0]
                linear_iterations[label] += ksp.getIterationNumber()

            problem.solver.getKSP().setPreSolve(before)
            problem.solver.getKSP().setPostSolve(after)
            problems.append(problem)
        pu = problems[0]
        trial_d = ufl.TrialFunction(D)
        a_damage = ((m["Gc"]/m["ell"]+2*(1-m["residual_stiffness"])*H)*trial_d*vd
                    + m["Gc"]*m["ell"]*ufl.inner(ufl.grad(trial_d), ufl.grad(vd)))*dx
        L_damage = 2*(1-m["residual_stiffness"])*H*vd*dx
        pd = BoundedDamageProblem(a_damage, L_damage, d, config, timings, linear_iterations)
        if s["preconditioner"] == "gamg" and s["linear_solver"] != "direct":
            coordinates = PETSc.Vec().createWithArray(D.tabulate_dof_coordinates()[:, :2].ravel().copy(), bsize=2, comm=comm)
            nullspace = PETSc.NullSpace().createRigidBody(coordinates)
            pu.A.setNearNullSpace(nullspace)
            coordinates.destroy()
        lower, upper = pd.x.duplicate(), pd.x.duplicate()
        lower.set(0)
        upper.set(1)
        pd.solver.setVariableBounds(lower, upper)
        energy_expression = dolfinx.fem.Expression(psi_plus, Q.element.interpolation_points, jit_options=compilation)
        elastic_form = dolfinx.fem.form(elastic_density*dx, jit_options=compilation)
        fracture_form = dolfinx.fem.form(m["Gc"]/(2*m["ell"])*d**2*dx
                                       + m["Gc"]*m["ell"]/2*ufl.inner(ufl.grad(d), ufl.grad(d))*dx, jit_options=compilation)
        # Variational reaction: unlifted residual at constrained x DOFs.
        residual_form = dolfinx.fem.form(Fu, jit_options=compilation)
        gauge = np.zeros_like(u.x.array)
        gauge[dl], gauge[dr] = -0.5, 0.5
        dofs = {"displacement": V.dofmap.index_map.size_global*V.dofmap.index_map_bs,
                "damage": D.dofmap.index_map.size_global*D.dofmap.index_map_bs}
        dofs["total"] = sum(dofs.values())
        write_json(output / "metadata.json", {"dofs": dofs, "cells": mesh.topology.index_map(2).size_global,
                   "backend": backend, "dolfinx": dolfinx.__version__, "petsc": PETSc.Sys.getVersion(),
                   "scalar_dtype": str(np.dtype(PETSc.ScalarType)), "mpi_ranks": comm.size,
                   "python": platform.python_version(), "host": platform.node(),
                   "geometry_hash": geo["geometry_hash"], "mesh_size": config["mesh"]["size"],
                   "ell_over_h": m["ell"]/config["mesh"]["size"],
                   "boundary_conditions": "left ux=-separation/2, right ux=+separation/2; top/bottom/pore traction free; one uy gauge",
                   "solver_options_u": solver_options(config),
                   "damage_solver": "primal_dual_active_set_full_matrix",
                   "damage_kkt_tolerance": max(s["snes_atol"], 10*s["ksp_rtol"])})
        # Portable serial arrays complement standard ParaView output.
        field_dir = output / "fields"
        vtk = None
        if config["output"]["save_fields"]:
            field_dir.mkdir()
            topology, _, coordinates = dolfinx.plot.vtk_mesh(D)
            np.savez_compressed(field_dir / "mesh.npz", points=coordinates, triangles=topology.reshape(-1, 4)[:, 1:])
            vtk = dolfinx.io.VTKFile(comm, str(output / "solution.pvd"), "w")
        setup_s = time.perf_counter()-start_setup
        previous_u, previous_d, accepted_H = u.x.array.copy(), d.x.array.copy(), H.x.array.copy()
        records, frames, rejections = [], [], []
        elapsed_start = time.perf_counter()
        current, step, work, previous_force = 0.0, 0, 0.0, 0.0
        nominal_increment = load["maximum_separation"]/load["increments"]
        targets = load.get("targets", np.linspace(nominal_increment, load["maximum_separation"], load["increments"]).tolist())
        if not targets or not all(np.isfinite(v) and v >= 0 for v in targets):
            raise ValueError("Loading targets must be a nonempty finite nonnegative list")

        def save_frame(separation, step):
            t = time.perf_counter()
            # P1 scalar/vector spaces share the same nodal ordering on this mesh.
            name = f"step_{step:05d}.npz"
            np.savez_compressed(field_dir / name, displacement=u.x.array.reshape(-1, 2), damage=d.x.array)
            vtk.write_function([u, d], separation)
            frames.append({"step": step, "separation": separation, "file": name})
            timings["field_output_s"] += time.perf_counter()-t

        if vtk:
            save_frame(0.0, 0)
        try:
            for target in targets:
                direction = 1 if target >= current else -1
                increment = abs(target-current)
                while abs(target-current) > 1e-13:
                    attempt = current+direction*min(increment, abs(target-current))
                    left.value, right.value = -attempt/2, attempt/2
                    u.x.array[:], d.x.array[:], H.x.array[:] = previous_u, previous_d, accepted_H
                    u.x.scatter_forward(); d.x.scatter_forward(); H.x.scatter_forward()
                    lower.array[:] = previous_d
                    old_u, old_d = previous_u.copy(), previous_d.copy()
                    iter_before = linear_iterations.copy()
                    t_attempt = time.perf_counter()
                    failure = None
                    try:
                        for stagger in range(1, s["maximum_stagger_iterations"]+1):
                            t = time.perf_counter()
                            pu.solve()
                            synchronize(backend)
                            timings["displacement_s"] += time.perf_counter()-t
                            if pu.solver.getConvergedReason() <= 0:
                                raise RuntimeError(f"Displacement SNES reason {pu.solver.getConvergedReason()}")
                            t = time.perf_counter()
                            current_energy.interpolate(energy_expression)
                            H.x.array[:] = np.maximum(accepted_H, current_energy.x.array)
                            H.x.scatter_forward()
                            timings["history_s"] += time.perf_counter()-t
                            t = time.perf_counter()
                            pd.solve()
                            synchronize(backend)
                            timings["damage_s"] += time.perf_counter()-t
                            if pd.solver.getConvergedReason() <= 0:
                                raise RuntimeError(f"Damage active-set reason {pd.solver.getConvergedReason()}: {pd.failure_detail}")
                            if not np.isfinite(u.x.array).all() or not np.isfinite(d.x.array).all():
                                raise RuntimeError("Nonfinite solution")
                            if d.x.array.min() < -1e-8 or d.x.array.max() > 1+1e-8 or np.min(d.x.array-previous_d) < -1e-8:
                                raise RuntimeError("Damage bounds/irreversibility violated")
                            du = np.linalg.norm(u.x.array-old_u)/max(np.linalg.norm(u.x.array), 1e-12)
                            dd = np.max(np.abs(d.x.array-old_d))
                            old_u[:], old_d[:] = u.x.array, d.x.array
                            if stagger >= 2 and max(du, dd) < s["stagger_tolerance"]:
                                break
                        else:
                            raise RuntimeError("Staggered iteration limit")
                    except RuntimeError as exc:
                        failure = str(exc)
                    if failure:
                        rejections.append({"separation": attempt, "increment": increment,
                                           "reason": failure, "seconds": time.perf_counter()-t_attempt})
                        increment /= 2
                        if increment < load["minimum_increment"]:
                            raise RuntimeError(f"Minimum load increment reached: {failure}")
                        print(f"Retry at separation {current:g}: increment -> {increment:g}", flush=True)
                        continue
                    t = time.perf_counter()
                    residual = fp.assemble_vector(residual_form)
                    reaction_left, reaction_right = float(residual.array[dl].sum()), float(residual.array[dr].sum())
                    force = float(np.dot(residual.array, gauge))*m["thickness"]
                    residual.destroy()
                    elastic = float(dolfinx.fem.assemble_scalar(elastic_form))*m["thickness"]
                    fracture = float(dolfinx.fem.assemble_scalar(fracture_form))*m["thickness"]
                    work += 0.5*(force+previous_force)*(attempt-current)
                    previous_force = force
                    step += 1
                    records.append({"step": step, "separation": attempt, "force": force,
                                    "reaction_left": reaction_left*m["thickness"],
                                    "reaction_right": reaction_right*m["thickness"],
                                    "elastic_energy": elastic, "fracture_energy": fracture,
                                    "external_work": work, "energy_balance_error": work-elastic-fracture,
                                    "damage_max": float(d.x.array.max()), "damage_mean": float(d.x.array.mean()),
                                    "stagger_iterations": stagger, "du_relative": du, "dd_max": dd,
                                    "u_snes_residual": float(pu.solver.getFunctionNorm()),
                                    "damage_kkt_residual": float(pd.solver.getFunctionNorm()),
                                    "u_ksp_iterations": linear_iterations["u"]-iter_before["u"],
                                    "d_ksp_iterations": linear_iterations["d"]-iter_before["d"],
                                    "attempt_seconds": time.perf_counter()-t_attempt})
                    timings["diagnostics_s"] += time.perf_counter()-t
                    current = attempt
                    previous_u[:], previous_d[:], accepted_H[:] = u.x.array, d.x.array, H.x.array
                    if vtk and (step % config["output"]["field_stride"] == 0 or abs(current-target) < 1e-13):
                        save_frame(current, step)
                    print(f"{backend} step={step} separation={current:.6g} force={force:.6g} damage={d.x.array.max():.4f} stagger={stagger}", flush=True)
        finally:
            if vtk:
                vtk.close()
                write_json(output / "frames.json", frames)
            if records:
                with (output / "history.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0]))
                    writer.writeheader(); writer.writerows(records)
            write_json(output / "rejections.json", rejections)
        synchronize(backend)
        total_s = time.perf_counter()-elapsed_start
        np.savez_compressed(output / "final_state.npz", displacement=u.x.array, damage=d.x.array,
                            history=H.x.array, coordinates=D.tabulate_dof_coordinates())
        if config["output"]["save_linear_systems"]:
            # Tangent matrices at the final accepted state, assembled identically
            # on host for CPU/GPU replay. RHS=A*x_ref is explicitly manufactured.
            for label, p, bc in [("u", pu, bcs), ("d", pd, [])]:
                A = fp.assemble_matrix(p.J, bcs=bc)
                A.assemble()
                ia, ja, values = A.getValuesCSR()
                xref = np.sin(np.arange(A.getSize()[0])*0.173)+0.1*np.cos(np.arange(A.getSize()[0])*0.071)
                x, b = A.createVecRight(), A.createVecLeft()
                x.array[:] = xref
                A.mult(x, b)
                np.savez_compressed(output / f"linear_{label}.npz", indptr=ia, indices=ja, data=values,
                                    shape=A.getSize(), rhs=b.array.copy(), reference=xref,
                                    block_size=2 if label == "u" else 1, coordinates=D.tabulate_dof_coordinates()[:, :2])
                x.destroy(); b.destroy(); A.destroy()
        summary = {"status": "complete", "backend": backend, "dofs": dofs,
                   "geometry_hash": geo["geometry_hash"], "setup_seconds": setup_s,
                   "simulation_seconds": total_s, "compute_seconds": total_s-timings["field_output_s"],
                   "timings": timings, "accepted_steps": len(records), "rejected_steps": len(rejections),
                   "linear_iterations": linear_iterations, "final_damage_max": float(d.x.array.max()),
                   "timing_note": "linear_solve is nested in displacement/damage; assembly overlaps those totals. Compute excludes field output; setup/JIT and final exports separate."}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "complete"})
        lower.destroy(); upper.destroy()
        return summary
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "reason": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=Path("results/porous_cpu"))
    p.add_argument("--backend", choices=["cpu", "gpu"])
    p.add_argument("--mesh-size", type=float)
    p.add_argument("--increments", type=int)
    p.add_argument("--maximum-separation", type=float)
    p.add_argument("--no-fields", action="store_true")
    p.add_argument("--direct", action="store_true", help="CPU MUMPS reference")
    p.add_argument("--preconditioner", choices=["jacobi", "gamg"])
    p.add_argument("--image-mesh", type=Path, required=True, help="Folder containing specimen.msh and geometry.json from the SEM mesher")
    a = p.parse_args()
    c = load_config(a.config)
    for value, section, key in [(a.backend, "solver", "backend"), (a.mesh_size, "mesh", "size"),
                                (a.increments, "loading", "increments"),
                                (a.maximum_separation, "loading", "maximum_separation")]:
        if value is not None:
            c[section][key] = value
    if a.no_fields:
        c["output"]["save_fields"] = False
    if a.direct:
        c["solver"]["linear_solver"] = "direct"
    if a.preconditioner:
        c["solver"]["preconditioner"] = a.preconditioner
    summary = run_simulation(c, a.output, a.image_mesh)
    print(f"Completed {summary['dofs']['total']} DOFs in {summary['compute_seconds']:.3f} s")


if __name__ == "__main__":
    main()
