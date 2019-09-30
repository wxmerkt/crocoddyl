///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

ActionModelAbstract::ActionModelAbstract(StateAbstract& state, unsigned int const& nu, unsigned int const& nr)
    : nu_(nu),
      nr_(nr),
      state_(state),
      unone_(Eigen::VectorXd::Zero(nu)),
      u_lower_limit_(Eigen::VectorXd::Constant(nu, 1, -std::numeric_limits<double>::infinity())),
      u_upper_limit_(Eigen::VectorXd::Constant(nu, 1, std::numeric_limits<double>::infinity())),
      has_control_limits_(false) {}

ActionModelAbstract::~ActionModelAbstract() {}

void ActionModelAbstract::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void ActionModelAbstract::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

void ActionModelAbstract::quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                                      const Eigen::Ref<const Eigen::VectorXd>& x, unsigned int const& maxiter,
                                      const double& tol) {
  assert((u.size() == nu_ || nu_ == 0) && "u has wrong dimension");
  assert(x.size() == state_.get_nx() && "x has wrong dimension");

  unsigned int const& ndx = state_.get_ndx();
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndx);
  if (nu_ == 0) {
    // TODO(cmastalli): create a method for autonomous systems
  } else {
    Eigen::VectorXd du = Eigen::VectorXd::Zero(nu_);
    for (unsigned int i = 0; i < maxiter; ++i) {
      calcDiff(data, x, u);
      state_.diff(x, data->xnext, dx);
      du = -pseudoInverse(data->Fu) * data->Fx * dx;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}

boost::shared_ptr<ActionDataAbstract> ActionModelAbstract::createData() {
  return boost::make_shared<ActionDataAbstract>(this);
}

const unsigned int& ActionModelAbstract::get_nu() const { return nu_; }

const unsigned int& ActionModelAbstract::get_nr() const { return nr_; }

StateAbstract& ActionModelAbstract::get_state() const { return state_; }

const Eigen::VectorXd& ActionModelAbstract::get_u_lower_limit() const { return u_lower_limit_; }

const Eigen::VectorXd& ActionModelAbstract::get_u_upper_limit() const { return u_upper_limit_; }

void ActionModelAbstract::set_u_lower_limit(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
  assert(nu_ == u_in.size() && "Number of rows of u_in must match nu_");
  u_lower_limit_ = u_in;
  update_has_control_limits();
}

void ActionModelAbstract::set_u_upper_limit(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
  assert(nu_ == u_in.size() && "Number of rows of u_in must match nu_");
  u_upper_limit_ = u_in;
  update_has_control_limits();
}

bool const& ActionModelAbstract::get_has_control_limits() const { return has_control_limits_; }

void ActionModelAbstract::update_has_control_limits() {
  has_control_limits_ = u_lower_limit_.allFinite() && u_upper_limit_.allFinite();
}

}  // namespace crocoddyl
