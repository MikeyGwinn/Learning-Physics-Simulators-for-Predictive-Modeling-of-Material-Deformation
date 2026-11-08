# THIS IS THE ACTUAL PHASE FIELD FRACTURE script using Gunnars pore generator on the graphite material properties.

# H-451 graphite RVE, replicating Chakraborty, Sabharwall & Carroll (2016), Sec. 3 and 4.1, Figs. 3-5, Table 2.
# Same phase-field model as sent_paper.i; only material, geometry, loading and units change.
#
# Units: micrometres and MPa (the paper's RVE units). Force per unit thickness is MPa*um,
# so stress = resid_y / 1000 um is in MPa (Fig. 4 axis). Strain = top displacement / 1000 um.
# Table 2: E = 7.6 GPa = 7600 MPa, nu = 0.18, g_c = 10.68 MPa*um, porosity 0.2.
# Sec. 3: uniform h = 4 um, l = 10 um, eta = 1e-3 (paper writes "um/s"; read as s/um in um units).
# Pores: very low stiffness regions. The stiffness map (pores, smeared small pores) comes from
# rve_gen.py as a grid file read by PiecewiseMultilinear and applied as a prefactor on the elasticity tensor.
# Loading (Sec. 4.1, Case 1): top pulled in y, left and bottom on symmetry (roller) BCs, right face free.
# Assumed: loading rate 1 um/s (not stated), so pseudo-time t = top displacement in um.

seed  = 1
h     = 4.0        # um (paper: 4 um); use 8 for a quick test
Lrve  = 1000.0     # um
E     = 7600.0     # MPa
nu    = 0.18
gc    = 10.68      # MPa*um
ell   = 10.0       # um
visco = 1e-3       # paper's eta
kres  = 1e-6
u_end = 4.0        # um  (strain 0.004; paper's Fig. 4 fails by ~0.0035)
du    = 0.02       # um per step at most (strain 2e-5)

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmax = ${Lrve}
    ymax = ${Lrve}
    nx = ${fparse ceil(Lrve / h)}
    ny = ${fparse ceil(Lrve / h)}
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Functions]
  [efac]                                           # E/E0: 1 matrix, (1 - f) smeared cells, 1e-4 in pores
    type = PiecewiseMultilinear
    data_file = efactor_${seed}.txt
  []
[]

[Modules/PhaseField/Nonconserved/c]
  free_energy = F
  kappa = kappa_op
  mobility = L
[]

[Physics/SolidMechanics/QuasiStatic/mech]
  add_variables = true
  strain = SMALL                                   # plane strain
  additional_generate_output = 'stress_yy'
  save_in = 'resid_x resid_y'
[]

[AuxVariables]
  [resid_x]
  []
  [resid_y]
  []
  [efac_aux]                                       # stiffness map on the mesh, to check the pores landed right
    order = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [efac_aux]
    type = FunctionAux
    variable = efac_aux
    function = efac
    execute_on = initial
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
  [pull_top]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = 't'
  []
  [sym_bottom]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [sym_left]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
[]

[Materials]
  [pfbulkmat]
    type = GenericConstantMaterial
    prop_names = 'gc_prop l visco'
    prop_values = '${gc} ${ell} ${visco}'
  []
  [define_mobility]
    type = ParsedMaterial
    enable_jit = false
    material_property_names = 'gc_prop visco'
    property_name = L
    expression = '1.0/(gc_prop * visco)'
  []
  [define_kappa]
    type = ParsedMaterial
    enable_jit = false
    material_property_names = 'gc_prop l'
    property_name = kappa_op
    expression = 'gc_prop * l'
  []
  [elasticity_tensor]
    type = ComputeElasticityTensor
    fill_method = symmetric_isotropic_E_nu
    C_ijkl = '${E} ${nu}'
    elasticity_tensor_prefactor = efac             # pores and smeared porosity scale the stiffness here
  []
  [damage_stress]
    type = ComputeLinearElasticPFFractureStress
    c = c
    E_name = 'elastic_energy'
    D_name = 'degradation'
    F_name = 'local_fracture_energy'
    decomposition_type = strain_spectral
  []
  [degradation]
    type = DerivativeParsedMaterial
    enable_jit = false
    property_name = degradation
    coupled_variables = 'c'
    expression = '(1.0-c)^2*(1.0 - eta) + eta'
    constant_names       = 'eta'
    constant_expressions = '${kres}'
    derivative_order = 2
  []
  [local_fracture_energy]
    type = DerivativeParsedMaterial
    enable_jit = false
    property_name = local_fracture_energy
    coupled_variables = 'c'
    material_property_names = 'gc_prop l'
    expression = 'c^2 * gc_prop / 2 / l'
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
  [top_disp]
    type = FunctionValuePostprocessor
    function = 't'
  []
  [resid_y]
    type = NodalSum
    variable = resid_y
    boundary = top
  []
  [max_c]
    type = ElementExtremeValue
    variable = c
  []
  [stress]                                         # MPa (Fig. 4 y-axis)
    type = ParsedPostprocessor
    pp_names = 'resid_y'
    expression = 'resid_y / ${Lrve}'
  []
  [strain]                                         # Fig. 4 x-axis
    type = ParsedPostprocessor
    pp_names = 'top_disp'
    expression = 'top_disp / ${Lrve}'
  []
  [damaged_area]                                   # um^2 of material with c > 0.9, a crack-length proxy
    type = ElementIntegralVariablePostprocessor
    variable = cracked
  []
[]

[AuxVariables]
  [cracked]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [cracked]
    type = ParsedAux
    variable = cracked
    coupled_variables = 'c'
    expression = 'if(c > 0.9, 1, 0)'
    enable_jit = false
    execute_on = 'timestep_end'
  []
[]

[VectorPostprocessors]
  [field]                                          # nodal damage every 10th step, for maps and animation
    type = NodalValueSampler
    variable = 'c'
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
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-10
  l_max_its = 50
  nl_max_its = 15
  [TimeStepper]
    type = IterationAdaptiveDT
    dt = ${du}
    optimal_iterations = 8
    growth_factor = 1.25
    cutback_factor = 0.5
  []
  dtmin = 1e-8
  dtmax = ${du}
  end_time = ${u_end}
[]

[Outputs]
  print_linear_residuals = false
  exodus = true
  csv = true
  [snap]
    type = CSV
    time_step_interval = 10
    time_data = true
    execute_postprocessors_on = none
  []
[]
