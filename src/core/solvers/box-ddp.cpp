///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, CNRS-LAAS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/core/solvers/box-ddp.hpp"

namespace crocoddyl {

template <class UnderlyingSolver>
AbstractControlLimitedDDPSolver<UnderlyingSolver>::AbstractControlLimitedDDPSolver(ShootingProblem& problem) : UnderlyingSolver(problem) {
  allocateData();

  const unsigned int& n_alphas = 10;
  SolverDDP::alphas_.resize(n_alphas);
  for (unsigned int n = 0; n < n_alphas; ++n) {
    SolverDDP::alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
}

template <class UnderlyingSolver>
AbstractControlLimitedDDPSolver<UnderlyingSolver>::~AbstractControlLimitedDDPSolver() {}

template <class UnderlyingSolver>
void AbstractControlLimitedDDPSolver<UnderlyingSolver>::allocateData() {
  UnderlyingSolver::allocateData();

  unsigned int nu_max = 0;
  unsigned int const& T = SolverAbstract::problem_.get_T();
  Quu_inv_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = SolverAbstract::problem_.running_models_[t];
    unsigned int const& nu = model->get_nu();

    // Store the largest number of controls across all models to allocate u_ll_, u_hl_
    if (nu > nu_max) nu_max = nu;

    Quu_inv_[t] = Eigen::MatrixXd::Zero(nu, nu);
  }

  u_ll_.resize(nu_max);
  u_hl_.resize(nu_max);
}

template <class UnderlyingSolver>
void AbstractControlLimitedDDPSolver<UnderlyingSolver>::computeGains(const unsigned int& t) {
  if (SolverAbstract::problem_.running_models_[t]->get_nu() > 0) {
    if (!SolverAbstract::problem_.running_models_[t]->get_has_control_limits()) {
      // No control limits on this model: Use vanilla DDP
      UnderlyingSolver::computeGains(t);
      return;
    }

    u_ll_ = SolverAbstract::problem_.running_models_[t]->get_u_lb() - SolverAbstract::us_[t];
    u_hl_ = SolverAbstract::problem_.running_models_[t]->get_u_ub() - SolverAbstract::us_[t];

    BoxQPSolution boxqp_sol = BoxQP(SolverDDP::Quu_[t], SolverDDP::Qu_[t], u_ll_, u_hl_, SolverAbstract::us_[t], 0.1, 100, 1e-5, SolverAbstract::ureg_);

    Quu_inv_[t].setZero();
    for (size_t i = 0; i < boxqp_sol.free_idx.size(); ++i)
      for (size_t j = 0; j < boxqp_sol.free_idx.size(); ++j)
        Quu_inv_[t](boxqp_sol.free_idx[i], boxqp_sol.free_idx[j]) = boxqp_sol.Hff_inv(i, j);

    // Compute controls
    SolverDDP::K_[t].noalias() = Quu_inv_[t] * SolverDDP::Qxu_[t].transpose();
    SolverDDP::k_[t].noalias() = -boxqp_sol.x;

    for (size_t j = 0; j < boxqp_sol.clamped_idx.size(); ++j) SolverDDP::K_[t](boxqp_sol.clamped_idx[j]) = 0.0;
  }
}

template <class UnderlyingSolver>
void AbstractControlLimitedDDPSolver<UnderlyingSolver>::forwardPass(const double& steplength) {
  assert(steplength <= 1. && "Step length has to be <= 1.");
  assert(steplength >= 0. && "Step length has to be >= 0.");
  SolverDDP::cost_try_ = 0.;
  SolverDDP::xnext_ = SolverAbstract::problem_.get_x0();
  unsigned int const& T = SolverAbstract::problem_.get_T();
  for (unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* m = SolverAbstract::problem_.running_models_[t];
    boost::shared_ptr<ActionDataAbstract>& d = SolverAbstract::problem_.running_datas_[t];
    if ((SolverAbstract::is_feasible_) || (steplength == 1)) {
      SolverDDP::xs_try_[t] = SolverDDP::xnext_;
    } else {
      m->get_state().integrate(SolverDDP::xnext_, SolverDDP::gaps_[t] * (steplength - 1), SolverDDP::xs_try_[t]);
    }
    m->get_state().diff(SolverAbstract::xs_[t], SolverDDP::xs_try_[t], SolverAbstract::xs_[t]);
    SolverDDP::us_try_[t].noalias() = SolverAbstract::us_[t] - SolverDDP::k_[t] * steplength - SolverDDP::K_[t] * SolverDDP::dx_[t];

    // Clamp!
    if (m->get_has_control_limits()) {
      SolverDDP::us_try_[t] = SolverDDP::us_try_[t].cwiseMax(m->get_u_lb()).cwiseMin(m->get_u_ub());
    }

    m->calc(d, SolverDDP::xs_try_[t], SolverDDP::us_try_[t]);
    SolverDDP::xnext_ = d->xnext;
    SolverDDP::cost_try_ += d->cost;

    if (raiseIfNaN(SolverDDP::cost_try_)) {
      throw "forward_error";
    }
    if (raiseIfNaN(SolverDDP::xnext_.lpNorm<Eigen::Infinity>())) {
      throw "forward_error";
    }
  }

  ActionModelAbstract* m = SolverAbstract::problem_.terminal_model_;
  boost::shared_ptr<ActionDataAbstract>& d = SolverAbstract::problem_.terminal_data_;

  if ((SolverAbstract::is_feasible_) || (steplength == 1)) {
    SolverDDP::xs_try_.back() = SolverDDP::xnext_;
  } else {
    m->get_state().integrate(SolverDDP::xnext_, SolverDDP::gaps_.back() * (steplength - 1), SolverDDP::xs_try_.back());
  }
  m->calc(d, SolverDDP::xs_try_.back());
  SolverDDP::cost_try_ += d->cost;

  if (raiseIfNaN(SolverDDP::cost_try_)) {
    throw "forward_error";
  }
}

template class AbstractControlLimitedDDPSolver<SolverDDP>;
template class AbstractControlLimitedDDPSolver<SolverFDDP>;

}  // namespace crocoddyl
