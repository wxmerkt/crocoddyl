///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

DifferentialActionModelAbstract::DifferentialActionModelAbstract(StateAbstract& state, unsigned int const& nu,
                                                                 unsigned int const& nr)
    : nu_(nu), nr_(nr), state_(state), unone_(Eigen::VectorXd::Zero(nu)) {}

DifferentialActionModelAbstract::~DifferentialActionModelAbstract() {}

void DifferentialActionModelAbstract::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void DifferentialActionModelAbstract::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelAbstract::createData() {
  return boost::make_shared<DifferentialActionDataAbstract>(this);
}

unsigned int const& DifferentialActionModelAbstract::get_nu() const { return nu_; }

unsigned int const& DifferentialActionModelAbstract::get_nr() const { return nr_; }

StateAbstract& DifferentialActionModelAbstract::get_state() const { return state_; }

const Eigen::VectorXd& DifferentialActionModelAbstract::get_u_lower_limit() const { return u_lower_limit_; }

const Eigen::VectorXd& DifferentialActionModelAbstract::get_u_upper_limit() const { return u_upper_limit_; }

void DifferentialActionModelAbstract::set_u_lower_limit(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
  assert(nu_ == u_in.size() && "Number of rows of u_in must match nu_");
  u_lower_limit_ = u_in;
  update_has_control_limits();
}

void DifferentialActionModelAbstract::set_u_upper_limit(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
  assert(nu_ == u_in.size() && "Number of rows of u_in must match nu_");
  u_upper_limit_ = u_in;
  update_has_control_limits();
}

bool const& DifferentialActionModelAbstract::get_has_control_limits() const { return has_control_limits_; }

void DifferentialActionModelAbstract::update_has_control_limits() {
  has_control_limits_ = u_lower_limit_.allFinite() && u_upper_limit_.allFinite();
}

}  // namespace crocoddyl
