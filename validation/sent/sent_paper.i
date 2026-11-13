# This is the SENT BENCHMARK (not the pore gen script

# SENT benchmark, replicating Chakraborty, Sabharwall & Carroll (2016), Sec. 2.1, Fig. 1, Table 1.
# Started from MOOSE's test modules/combined/test/tests/phase_field_fracture/crack2d_iso.i
#
# Paper setup (Fig. 1a): 1 mm x 1 mm plate, 0.5 mm edge notch at mid-height, bottom u = v = 0,
# top u = 0 with v pulled up. Bilinear quads, h < l/2 along the crack path, plane strain (2D SMALL).
# Table 1: E = 207 GPa, nu = 0.3, g_c = 0.0027 GPa*mm, l = 0.0075 mm, eta = 1e-4 s/mm, k = 1e-6.
# The notch is on the mid-plane, so we model the top half (symmetry): the bottom edge of this
# model is the crack plane, held in y only along the ligament. Full-specimen strain = 2*u_top / 1 mm.
#
# Units: mm, GPa, so force per unit thickness is GPa*mm and stress = resid_y / W is in GPa (Fig. 1b axis).
#
# Viscosity: MOOSE's c equation works out to
#   visco * dc/dt = l*lap(c) + 2(1-c) psi+ / g_c - c/l
# which is the paper's Eq. 9 term for term (without its <>_+ bracket), so visco = eta = 1e-4.
# Pseudo-time t = top displacement in mm (assumes 1 mm/s; the paper does not state its rate).
#
# Differences from the paper kept on purpose for now (see notes at the end of the file):
#   1. irreversibility: MOOSE uses a history field (max psi+) instead of Eq. 9's <>_+ bracket;
#   2. uniform mesh instead of the paper's graded mesh;
#   3. half model instead of the full plate.

# ---------- parameters ----------
W     = 1.0       # mm
H     = 0.5       # half height, mm (full plate 1 mm)
a     = 0.5       # notch length, mm
h     = 0.0035    # element size along x and at the crack plane y = 0, mm (paper: h < l/2 = 0.00375)
bias  = 1.0       # growth of element height away from the crack plane (1 = uniform; 1.04 = graded like the paper)
E     = 207.0     # GPa
nu    = 0.3
gc    = 0.0027    # GPa*mm
ell   = 0.0075    # mm
visco = 1e-4      # = eta (Table 1)
kres  = 1e-6      # k in Eq. 4b
u_end = 0.004     # top displacement, mm  (full strain 0.008, well past the paper's drop at ~0.0052)
du    = 1e-5      # first step, mm
[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmax = ${W}                         # CHANGED (was 1)
    ymax = ${H}                         # CHANGED (was 0.5)
    nx = ${fparse ceil(W / h)}          # CHANGED (was 20, h = 0.05 > l)
    ny = ${fparse if(bias > 1.0001, ceil(log(1 + H * (bias - 1) / h) / log(bias)), ceil(H / h))}
    bias_y = ${bias}                    # elements smallest at y = 0 (the crack plane), growing toward the top
  []
  [noncrack]
    type = BoundingBoxNodeSetGenerator
    new_boundary = noncrack
    bottom_left = '${a} 0 0'            # CHANGED (notch length is now a parameter)
    top_right = '${W} 0 0'
    input = gen
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Modules/PhaseField/Nonconserved/c]
  free_energy = F
  kappa = kappa_op
  mobility = L
[]

[Physics/SolidMechanics/QuasiStatic/mech]
  add_variables = true
  strain = SMALL                        # 2D SMALL = plane strain
  additional_generate_output = 'stress_yy'
  save_in = 'resid_x resid_y'
[]

[AuxVariables]
  [resid_x]
  []
  [resid_y]
  []
[]

[Kernels]
  [solid_x]
    type = PhaseFieldFractureMechanicsOffDiag
    variable = disp_x
    component = 0
    c = c
  []
  [solid_y]
    type = PhaseFieldFractureMechanicsOffDiag
    variable = disp_y
    component = 1
    c = c
  []
[]

[BCs]
  [ydisp]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = 't'
  []
  [yfix]
    type = DirichletBC
    variable = disp_y
    boundary = noncrack
    value = 0
  []
  [xfix]
    type = DirichletBC
    variable = disp_x
    boundary = top
    value = 0
  []
[]

[Materials]
  [pfbulkmat]
    # L = 1/(gc*visco), kappa = gc*l  ->  visco*dc/dt = l lap c + 2(1-c)psi+/gc - c/l  (paper Eq. 9)
    type = GenericConstantMaterial
    prop_names = 'gc_prop l visco'
    prop_values = '${gc} ${ell} ${visco}'          # CHANGED (was 1e-3 0.04 1e-4)
  []
  [define_mobility]
    type = ParsedMaterial
    enable_jit = false   # container has no g++; skip JIT, use bytecode
    material_property_names = 'gc_prop visco'
    property_name = L
    expression = '1.0/(gc_prop * visco)'
  []
  [define_kappa]
    type = ParsedMaterial
    enable_jit = false   # container has no g++; skip JIT, use bytecode
    material_property_names = 'gc_prop l'
    property_name = kappa_op
    expression = 'gc_prop * l'
  []
  [elasticity_tensor]
    type = ComputeElasticityTensor
    fill_method = symmetric_isotropic_E_nu         # CHANGED (was Lame 120, 80)
    C_ijkl = '${E} ${nu}'                          # 207 GPa, 0.3 (Table 1)
  []
  [damage_stress]
    type = ComputeLinearElasticPFFractureStress
    c = c
    E_name = 'elastic_energy'
    D_name = 'degradation'
    F_name = 'local_fracture_energy'
    decomposition_type = strain_spectral           # Miehe split, as in the paper
  []
  [degradation]
    type = DerivativeParsedMaterial
    enable_jit = false   # container has no g++; skip JIT, use bytecode
    property_name = degradation
    coupled_variables = 'c'
    expression = '(1.0-c)^2*(1.0 - eta) + eta'
    constant_names       = 'eta'
    constant_expressions = '${kres}'               # k = 1e-6 (Table 1)
    derivative_order = 2
  []
  [local_fracture_energy]
    type = DerivativeParsedMaterial
    enable_jit = false   # container has no g++; skip JIT, use bytecode
    property_name = local_fracture_energy
    coupled_variables = 'c'
    material_property_names = 'gc_prop l'
    expression = 'c^2 * gc_prop / 2 / l'           # AT2
    derivative_order = 2
  []
  [fracture_driving_energy]
    type = DerivativeSumMaterial
    coupled_variables = c
    sum_materials = 'elastic_energy local_fracture_energy'
    derivative_order = 2
    property_name = F
  []
[]

[Postprocessors]
  [top_disp]                                       # CHANGED (added): x-axis of the curve
    type = FunctionValuePostprocessor
    function = 't'
  []
  [resid_x]
    type = NodalSum
    variable = resid_x
    boundary = top
  []
  [resid_y]                                        # reaction force, N per mm thickness
    type = NodalSum
    variable = resid_y
    boundary = top
  []
  [max_c]                                          # CHANGED (added): when the crack starts
    type = ElementExtremeValue
    variable = c
  []
  [stress]                                         # nominal stress, GPa (Fig. 1b y-axis)
    type = ParsedPostprocessor
    pp_names = 'resid_y'
    expression = 'resid_y / ${W}'
  []
  [strain]                                         # full-specimen strain (Fig. 1b x-axis)
    type = ParsedPostprocessor
    pp_names = 'top_disp'
    expression = '2 * top_disp / (2 * ${H})'
  []
[]

[VectorPostprocessors]
  [damage_line]                                    # Fig. 1d: c across the crack, every step
    type = LineValueSampler
    variable = c
    start_point = '0.75 0 0'
    end_point = '0.75 0.06 0'
    num_points = 241
    sort_by = y
    outputs = csv
  []
  [field]                                          # full nodal field, every 10th step (for maps and animation)
    type = NodalValueSampler
    variable = 'c disp_x disp_y'
    sort_by = id
    outputs = snap
  []
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  solve_type = PJFNK
  petsc_options_iname = '-pc_type -ksp_gmres_restart -sub_ksp_type -sub_pc_type -pc_asm_overlap'
  petsc_options_value = 'asm      31                  preonly       lu           1'

  nl_rel_tol = 1e-6                                # paper: 1e-4 per field; a little tighter here
  nl_abs_tol = 1e-12
  l_max_its = 50
  nl_max_its = 15
  [TimeStepper]                                    # paper: halve the step when a solve fails
    type = IterationAdaptiveDT
    dt = ${du}
    optimal_iterations = 8
    growth_factor = 1.25
    cutback_factor = 0.5
  []
  dtmin = 1e-10
  dtmax = 2e-5
  end_time = ${u_end}
[]

[Outputs]
  print_linear_residuals = false                   # belongs here, not under [console]
  exodus = true
  csv = true                                       # postprocessors + damage line, every step
  [snap]
    type = CSV
    time_step_interval = 10
    time_data = true                               # writes *_field_time.csv: which time each snapshot is
    execute_postprocessors_on = none
  []
  [console]
    type = Console
    max_rows = 5
  []
[]

# NOTES ON REMAINING DIFFERENCES FROM THE PAPER
# 1. Eq. 9 enforces dc/dt >= 0 with a Macaulay bracket. MOOSE's ComputeLinearElasticPFFractureStress
#    instead drives damage with a history field (largest psi+ so far). Both stop healing; they can
#    differ slightly after the peak. Matching the bracket exactly needs a variational-inequality
#    solve (PETSc vinewtonrsls with c bounded below by its old value) or a custom kernel.
# 2. The paper refines the mesh only near the crack path; this file uses a uniform h < l/2.
