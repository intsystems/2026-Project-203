import EF21Muon
import Lean

/-! Reject unexpected axioms throughout the imported EF21Muon library, including
private declarations. Select declarations by their defining module rather than
their namespace, so generated names and declarations in other namespaces are
also checked. The individual reports below identify the main paper results. -/

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let mut checked : Nat := 0
  for (decl, _) in env.constants.toList do
    let some idx := env.getModuleIdxFor? decl | continue
    let moduleName := env.header.moduleNames[idx.toNat]!
    unless (`EF21Muon).isPrefixOf moduleName do continue
    let axioms ← collectAxioms decl
    let unexpected := axioms.filter fun ax => !allowed.contains ax
    unless unexpected.isEmpty do
      throwError "Axiom audit failed for {decl}: unexpected axioms {unexpected}"
    checked := checked + 1
  if checked == 0 then
    throwError "Axiom audit failed: no EF21Muon declarations were checked"
  logInfo m!"Axiom audit passed for {checked} declarations defined in EF21Muon modules. Allowed axioms: {allowed}"

#print axioms EF21Muon.Compression.scaledSign_error_identity
#print axioms EF21Muon.Compression.scaledSign_contractivity
#print axioms EF21Muon.Compression.scaledSign_euclidean_contracts
#print axioms EF21Muon.Compression.scaledSign_realized_identity
#print axioms EF21Muon.Compression.one_sparse_realizedContraction
#print axioms EF21Muon.Compression.constant_magnitude_exact
#print axioms EF21Muon.Compression.denseVector_pos
#print axioms EF21Muon.Compression.denseVector_contraction_tendsto
#print axioms EF21Muon.Compression.effective_support_contraction
#print axioms EF21Muon.Compression.realizedContraction_iid_ae
#print axioms EF21Muon.Compression.integral_abs_centered_gaussian
#print axioms EF21Muon.Compression.gaussian_contraction_moment_ratio
#print axioms EF21Muon.Compression.realizedContraction_gaussian_ae
#print axioms EF21Muon.Probability.rareLaw_probability
#print axioms EF21Muon.Probability.rareLaw_moments
#print axioms EF21Muon.Probability.rareLaw_first_moment_tendsto
#print axioms EF21Muon.Probability.rareLaw_second_moment_not_tendsto
#print axioms EF21Muon.Distributed.zeta_le_rateEnvelope
#print axioms EF21Muon.Distributed.rate_step_condition
#print axioms EF21Muon.Distributed.Problem.smooth_rate
#print axioms EF21Muon.Distributed.smooth_rate_isBigO
#print axioms EF21Muon.Distributed.Problem.horizon_rate_isBigO
#print axioms EF21Muon.Distributed.matrix_horizon_rate_isBigO
#print axioms EF21Muon.Distributed.Problem.weighted_smooth_rate
#print axioms EF21Muon.Distributed.layerRateGamma_valid
#print axioms EF21Muon.Distributed.Problem.layer_horizon_cost
#print axioms EF21Muon.Distributed.Problem.weighted_horizon_rate_isBigO
#print axioms EF21Muon.Distributed.matrix_weighted_horizon_rate_isBigO
#print axioms EF21Muon.Distributed.generalizedRadius_sq
#print axioms EF21Muon.Distributed.generalized_exponent_conditions
#print axioms EF21Muon.Distributed.smooth_upper_at
#print axioms EF21Muon.Distributed.generalized_gradient_continuous
#print axioms EF21Muon.Distributed.generalized_smooth_upper
#print axioms EF21Muon.Distributed.generalized_radius_descent
#print axioms EF21Muon.Distributed.generalized_tracking_descent
#print axioms EF21Muon.Distributed.generalized_gradient_self_bound
#print axioms EF21Muon.Distributed.generalized_weighted_gradient_self_bound
#print axioms EF21Muon.Probability.integral_norm_sq_le
#print axioms EF21Muon.Probability.noise_ema_variance
#print axioms EF21Muon.Probability.debiased_momentum_identity
#print axioms EF21Muon.Distributed.radius_local_recurrence
#print axioms EF21Muon.Distributed.radius_compression_recurrence
#print axioms EF21Muon.Distributed.radius_noise_recurrence
#print axioms EF21Muon.Distributed.radiusEnergy_recurrence
#print axioms EF21Muon.Probability.accumulated_cost_geometric_bound
#print axioms EF21Muon.Probability.expectedMinimum_le_average
#print axioms EF21Muon.Distributed.GeneralizedProblem.radius_drift
#print axioms EF21Muon.Distributed.GeneralizedProblem.radius_finite_time
#print axioms EF21Muon.Distributed.generalizedHorizonControls_paper
#print axioms EF21Muon.Distributed.GeneralizedProblem.generalized_total_cost_for
#print axioms EF21Muon.Distributed.GeneralizedProblem.generalized_average_rate
#print axioms EF21Muon.Distributed.GeneralizedProblem.generalized_minimum_rate
#print axioms EF21Muon.Distributed.GeneralizedProblem.integral_weightedNuclear
#print axioms EF21Muon.Distributed.matrix_generalized_horizon_rate_rpow
#print axioms EF21Muon.Distributed.matrix_generalized_unweighted_rate
#print axioms EF21Muon.Distributed.radius_run_erases
#print axioms EF21Muon.Distributed.radius_pathState_erases
#print axioms EF21Muon.Distributed.Correspondence.ofRadiusScheme_round
#print axioms EF21Muon.Distributed.radiusPaperNext_eq_phases
#print axioms EF21Muon.Distributed.radiusPaper_pathState
#print axioms EF21Muon.Distributed.GeneralizedProblem.generalized_paper_iterate_integrable
#print axioms EF21Muon.Distributed.GeneralizedProblem.generalized_paper_expectation_eq
#print axioms EF21Muon.Distributed.GeneralizedProblem.paperRadiusMinimum_le
#print axioms EF21Muon.Distributed.matrix_generalized_paper_rate
#print axioms EF21Muon.Distributed.matrix_generalized_paper_unweighted_rate
#print axioms EF21Muon.Nesterov.target_increment
#print axioms EF21Muon.Nesterov.compression_recurrence
#print axioms EF21Muon.Nesterov.smooth_finite_time
#print axioms EF21Muon.Nesterov.smooth_rate
#print axioms EF21Muon.Nesterov.matrix_horizon_rate_isBigO
#print axioms EF21Muon.Nesterov.Correspondence.exact_instance
#print axioms EF21Muon.Nesterov.Correspondence.ofScheme_round
#print axioms EF21Muon.Nesterov.sharpPaperSum_eq
#print axioms EF21Muon.Nesterov.paperSmoothAverage_eq
#print axioms EF21Muon.Nesterov.smooth_paper_rate
#print axioms EF21Muon.Nesterov.paper_horizon_rate_isBigO
#print axioms EF21Muon.Nesterov.radiusEnergy_recurrence
#print axioms EF21Muon.Nesterov.generalized_finite_time
#print axioms EF21Muon.Nesterov.generalized_total_cost_for
#print axioms EF21Muon.Nesterov.matrix_generalized_horizon_rate_rpow
#print axioms EF21Muon.Nesterov.radiusPaper_pathState
#print axioms EF21Muon.Nesterov.generalized_paper_iterate_integrable
#print axioms EF21Muon.Nesterov.generalized_paper_expectation_eq
#print axioms EF21Muon.Nesterov.matrix_generalized_paper_rate
#print axioms EF21Muon.Nesterov.matrix_generalized_paper_unweighted_rate
#print axioms EF21Muon.LongTails.accuracyBeta_noise
#print axioms EF21Muon.LongTails.accuracyBudget_zero
#print axioms EF21Muon.LongTails.ema_accuracy
#print axioms EF21Muon.LongTails.nesterov_accuracy
#print axioms EF21Muon.LongTails.ema_polynomial_bound
#print axioms EF21Muon.LongTails.nesterov_paper_polynomial_bound
#print axioms EF21Muon.LongTails.Profile.optimalBeta_minimizes
#print axioms EF21Muon.LongTails.Profile.optimalBudget_minimal
#print axioms EF21Muon.LongTails.Profile.certificate_barrier
#print axioms EF21Muon.LongTails.ema_optimalBudget_le_previous
#print axioms EF21Muon.LongTails.nesterov_optimalBudget_le_previous
#print axioms EF21Muon.Theorems.Optimized.ema
#print axioms EF21Muon.Theorems.Optimized.nesterov
#print axioms EF21Muon.Theorems.Optimized.nesterov_paper
#print axioms EF21Muon.LongTails.execute_batchProtocol
#print axioms EF21Muon.LongTails.worker_query_count
#print axioms EF21Muon.Theorems.StochasticEMA.result
#print axioms EF21Muon.Theorems.StochasticNesterov.result
#print axioms EF21Muon.Distributed.Correspondence.exact_instance
#print axioms EF21Muon.Distributed.Correspondence.models_exact
#print axioms EF21Muon.Distributed.Correspondence.aggregate_update
#print axioms EF21Muon.Distributed.Correspondence.observable_sum
#print axioms EF21Muon.Distributed.Correspondence.average_transfer
#print axioms EF21Muon.Distributed.Correspondence.ofScheme_round
#print axioms EF21Muon.tracking_step
#print axioms EF21Muon.muonUSign_stationarity
#print axioms EF21Muon.muonSign_stationarity
#print axioms EF21Muon.stationarity_at_budget
#print axioms EF21Muon.execute_protocol
#print axioms EF21Muon.protocol_stationarity
#print axioms EF21Muon.Theorems.DeterministicMuonUSign.result
#print axioms EF21Muon.Theorems.DeterministicMuonSign.result
#print axioms EF21Muon.Stochastic.Sampling.variance_identity
#print axioms EF21Muon.Stochastic.expectedTotal_eq_paths
#print axioms EF21Muon.Stochastic.nuclearNorm_le_sqrt_rank
#print axioms EF21Muon.Stochastic.polar_frobenius_le_sqrt_rank
#print axioms EF21Muon.Stochastic.momentum_rec
#print axioms EF21Muon.Stochastic.uplink_rec
#print axioms EF21Muon.Stochastic.downlink_rec
#print axioms EF21Muon.Stochastic.Parameters.zeta_eq_paper
#print axioms EF21Muon.Stochastic.Parameters.zetaWith_one_eq_paper
#print axioms EF21Muon.Stochastic.Parameters.zetaWith_zero_le_one
#print axioms EF21Muon.Stochastic.Parameters.zetaWith_zero_lt_one
#print axioms EF21Muon.Stochastic.potential_descent
#print axioms EF21Muon.Stochastic.average_stationarity
#print axioms EF21Muon.Stochastic.muonUSign_stationarity
#print axioms EF21Muon.Stochastic.muonSign_stationarity
#print axioms EF21Muon.Stochastic.muonUSign_tighter_stationarity
#print axioms EF21Muon.Probability.affine_variance_identity
#print axioms EF21Muon.Probability.independent_average_variance
#print axioms EF21Muon.Probability.totalPotential_integral_bound
#print axioms EF21Muon.Probability.momentum_recurrence
#print axioms EF21Muon.Probability.uplink_recurrence
#print axioms EF21Muon.Distributed.global_recurrence
#print axioms EF21Muon.Distributed.local_recurrence
#print axioms EF21Muon.Distributed.compression_recurrence
#print axioms EF21Muon.Distributed.smooth_upper
#print axioms EF21Muon.Distributed.layer_descent
#print axioms EF21Muon.Distributed.zeta_eq_paper
#print axioms EF21Muon.Distributed.exact_downlink_le
#print axioms EF21Muon.Distributed.Problem.finite_time
#print axioms EF21Muon.Distributed.Problem.average_stationarity
#print axioms EF21Muon.Distributed.Problem.initial_potential
#print axioms EF21Muon.Distributed.Problem.uniform_cost
#print axioms EF21Muon.Distributed.explicitControls_beta
#print axioms EF21Muon.Distributed.explicitControls_gamma
#print axioms EF21Muon.Distributed.Problem.stationarity_at_budget
#print axioms EF21Muon.Distributed.matrix_stationarity
#print axioms EF21Muon.Distributed.nuclear_variance_suffices
#print axioms EF21Muon.Divergence.polar_of_orthogonal_posDef
#print axioms EF21Muon.Divergence.polar_smul_pos
#print axioms EF21Muon.Divergence.linearObjective_contDiff
#print axioms EF21Muon.Divergence.linearObjective_gradient
#print axioms EF21Muon.Divergence.linearObjective_spectralSmooth
#print axioms EF21Muon.Divergence.effectiveMomentum_eq
#print axioms EF21Muon.Divergence.linear_objective_increment
#print axioms EF21Muon.Divergence.linear_objective_strictAscent
#print axioms EF21Muon.Divergence.linear_objective_tendsto_atTop
#print axioms EF21Muon.Divergence.SignMuon.polar_gradient
#print axioms EF21Muon.Divergence.SignMuon.pairing
#print axioms EF21Muon.Divergence.SignMuon.ascent
#print axioms EF21Muon.Divergence.SignMuon.divergence
#print axioms EF21Muon.Divergence.MuonSign.polar_pattern
#print axioms EF21Muon.Divergence.MuonSign.uplink_pairing
#print axioms EF21Muon.Divergence.MuonSign.bidirectional_pairing
#print axioms EF21Muon.Divergence.MuonSign.ascent
#print axioms EF21Muon.Divergence.MuonSign.divergence
#print axioms EF21Muon.Divergence.EF21SignMuon.harmless_cycle
#print axioms EF21Muon.Divergence.EF21SignMuon.harmless_average
#print axioms EF21Muon.Divergence.EF21SignMuon.wrong_sign_cycle
#print axioms EF21Muon.Divergence.EF21SignMuon.wrong_sign_average
#print axioms EF21Muon.Divergence.EF21SignMuon.position_tail
#print axioms EF21Muon.Divergence.EF21SignMuon.prescribed_polar
#print axioms EF21Muon.Divergence.EF21SignMuon.rawObjective_gradient
#print axioms EF21Muon.Divergence.plateauDerivative_plus
#print axioms EF21Muon.Divergence.plateauDerivative_minus
#print axioms EF21Muon.Divergence.plateauDerivative_mean_zero
#print axioms EF21Muon.Divergence.plateauPotential_periodic
#print axioms EF21Muon.Divergence.EF21SignMuon.derivative₁_plateau_plus
#print axioms EF21Muon.Divergence.EF21SignMuon.derivative₁_plateau_minus
#print axioms EF21Muon.Divergence.EF21SignMuon.derivative₂_plateau_plus
#print axioms EF21Muon.Divergence.EF21SignMuon.derivative₂_plateau_minus
#print axioms EF21Muon.Divergence.EF21SignMuon.plateau_realization
#print axioms EF21Muon.Divergence.EF21SignMuon.normalized_objective
#print axioms EF21Muon.Divergence.EF21SignMuon.scale_reduction
#print axioms EF21Muon.Divergence.EF21SignMuon.normalized_run
#print axioms EF21Muon.Divergence.EF21SignMuon.divergence

#print axioms EF21Muon.Nesterov.Averaged.smooth_finite_time
#print axioms EF21Muon.LongTails.averaged_nesterov_polynomial_bound
#print axioms EF21Muon.LongTails.averaged_nesterov_paper_polynomial_bound
#print axioms EF21Muon.LongTails.averaged_nesterov_coefficients_le
#print axioms EF21Muon.LongTails.averaged_nesterov_optimalBudget_le
#print axioms EF21Muon.Theorems.WorkerAveraged.nesterov
#print axioms EF21Muon.Theorems.WorkerAveraged.nesterov_paper

#print axioms EF21Muon.Experiments.radius_descent
#print axioms EF21Muon.Experiments.scheduled_stationarity
#print axioms EF21Muon.Experiments.scheduled_residual
#print axioms EF21Muon.Experiments.tracking_envelope
#print axioms EF21Muon.Experiments.query_gradient_transfer
#print axioms EF21Muon.Experiments.fixed_momentum_error_sum
#print axioms EF21Muon.Experiments.fixed_momentum_stationarity
#print axioms EF21Muon.Experiments.filtered_noise_bound
#print axioms EF21Muon.Experiments.noise_variance_cap
#print axioms EF21Muon.Experiments.radius_bias_bound
#print axioms EF21Muon.Experiments.target_increment_bound

#print axioms EF21Muon.Experiments.Trajectory.tracking
#print axioms EF21Muon.Experiments.Trajectory.expected_stationarity
#print axioms EF21Muon.Experiments.Trajectory.MartingaleInnovations.past_orthogonal
#print axioms EF21Muon.Experiments.Trajectory.fresh_stationarity

#print axioms EF21Muon.Experiments.BlockTrajectory.Problem.ofDistributed
#print axioms EF21Muon.Experiments.BlockTrajectory.advance_block
#print axioms EF21Muon.Experiments.BlockTrajectory.run_block
#print axioms EF21Muon.Experiments.BlockTrajectory.tracking
#print axioms EF21Muon.Experiments.BlockTrajectory.expected_stationarity_sum
#print axioms EF21Muon.Experiments.BlockTrajectory.fresh_stationarity

#print axioms EF21Muon.RandomTies.coin_fair
#print axioms EF21Muon.RandomTies.coin_independence
#print axioms EF21Muon.RandomTies.stream_fair
#print axioms EF21Muon.RandomTies.stream_independence
#print axioms EF21Muon.RandomTies.compressor_error_sq
#print axioms EF21Muon.Divergence.RandomTies.constant_direction
#print axioms EF21Muon.Divergence.RandomTies.run_eq
#print axioms EF21Muon.Divergence.EF21SignMuon.positionWith_eq
#print axioms EF21Muon.RandomTies.ema_finite_time
#print axioms EF21Muon.RandomTies.nest_finite_time
#print axioms EF21Muon.RandomTies.ema_generalized_finite_time
#print axioms EF21Muon.RandomTies.nest_generalized_finite_time
#print axioms EF21Muon.RandomTies.ema_paper_average
#print axioms EF21Muon.RandomTies.nest_paper_average
#print axioms EF21Muon.RandomTies.ema_generalized_paper_minimum_rate
#print axioms EF21Muon.RandomTies.nest_generalized_paper_minimum_rate
#print axioms EF21Muon.RandomTies.ema_generalized_paper_iterate_integrable
#print axioms EF21Muon.RandomTies.nest_generalized_paper_iterate_integrable
#print axioms EF21Muon.RandomTies.ema_generalized_paper_expectation_eq
#print axioms EF21Muon.RandomTies.nest_generalized_paper_expectation_eq
#print axioms EF21Muon.Experiments.RandomizedBlocks.fresh_stationarity
#print axioms EF21Muon.RandomTies.experimental_trackingBudget_eq
