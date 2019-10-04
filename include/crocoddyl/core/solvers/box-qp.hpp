///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
//
// This file was originally part of Exotica, cf.
// https://github.com/ipab-slmc/exotica/blob/master/exotica_core/include/exotica_core/tools/box_qp.h
//
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <vector>

#include <iostream>
#include "crocoddyl/core/solvers/timer.h"

namespace crocoddyl {
struct BoxQPData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BoxQPData(const int nu)
      : Hff(nu, nu),
        Hfc(nu, nu),
        Hff_inv(nu, nu),
        Hff_inv_llt(nu),
        I(nu, nu),
        x(nu),
        delta_xf(nu),
        grad(nu),
        q_free(nu),
        x_free(nu),
        x_clamped(nu),
        x_new(nu),
        x_diff(nu) {
    free_idx.reserve(nu);
    clamped_idx.reserve(nu);
  }

  void reset() {
    Hff_inv.setIdentity();
    it = 0;
    free_idx.clear();
    clamped_idx.clear();
  }

  int it = 0;
  Eigen::MatrixXd Hff;
  Eigen::MatrixXd Hfc;
  Eigen::MatrixXd Hff_inv;
  Eigen::MatrixXd I;
  Eigen::VectorXd x;
  Eigen::VectorXd delta_xf;
  Eigen::VectorXd grad;
  Eigen::VectorXd q_free;
  Eigen::VectorXd x_free;
  Eigen::VectorXd x_clamped;
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt;
  std::vector<size_t> free_idx;
  std::vector<size_t> clamped_idx;
  double f_old;
  double f_new;
  bool armijo_reached;
  Eigen::VectorXd x_new;
  Eigen::VectorXd x_diff;
};

// Based on Yuval Tassa's BoxQP
// Cf. https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
void BoxQP(BoxQPData& data, const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& b_low,
           const Eigen::VectorXd& b_high, const Eigen::VectorXd& x_init, const double gamma, const int max_iterations,
           const double epsilon, const double lambda) {
  data.reset();
  data.x = x_init;
  data.grad = q + H * x_init;

  data.Hff_inv_llt.compute(data.I * lambda + H);
  data.Hff_inv_llt.solveInPlace(data.Hff_inv);

  if (data.grad.lpNorm<Eigen::Infinity>() <= epsilon) {
    return;  // {data.Hff_inv, x_init, {}, {}, 0};
  }

  while (data.grad.lpNorm<Eigen::Infinity>() > epsilon && data.it < max_iterations) {
    ++data.it;
    data.grad.noalias() = q + H * data.x;
    data.clamped_idx.clear();
    data.free_idx.clear();

    for (int i = 0; i < data.grad.size(); ++i) {
      if ((data.x(i) == b_low(i) && data.grad(i) > 0) || (data.x(i) == b_high(i) && data.grad(i) < 0)) {
        data.clamped_idx.push_back(i);
      } else {
        data.free_idx.push_back(i);
      }
    }

    if (data.free_idx.size() == 0) {
      return;  // {data.Hff_inv, data.x, data.free_idx, data.clamped_idx, data.it};
    }

    // data.Hff.resize(data.free_idx.size(), free_idx.size());
    // data.Hfc.resize(data.free_idx.size(), clamped_idx.size());

    if (data.clamped_idx.size() == 0) {
      data.Hff = H;
    } else {
      for (size_t i = 0; i < data.free_idx.size(); ++i) {
        for (size_t j = 0; j < data.free_idx.size(); ++j) {
          data.Hff(i, j) = H(data.free_idx[i], data.free_idx[j]);
        }
      }

      for (size_t i = 0; i < data.free_idx.size(); ++i) {
        for (size_t j = 0; j < data.clamped_idx.size(); ++j) {
          data.Hfc(i, j) = H(data.free_idx[i], data.clamped_idx[j]);
        }
      }
    }

    // NOTE: Array indexing not supported in current eigen version
    // Eigen::VectorXd q_free(data.free_idx.size()), x_free(data.free_idx.size()), x_clamped(clamped_idx.size());
    for (size_t i = 0; i < data.free_idx.size(); ++i) {
      data.q_free(i) = q(data.free_idx[i]);
      data.x_free(i) = data.x(data.free_idx[i]);
    }

    for (size_t j = 0; j < data.clamped_idx.size(); ++j) {
      data.x_clamped(j) = data.x(data.clamped_idx[j]);
    }

    // The dimension of Hff has changed - reinitialise LLT
    // Hff_inv_llt = Eigen::LLT<Eigen::MatrixXd>(Hff.rows());
    // Hff_inv_llt.compute(Eigen::MatrixXd::Identity(Hff.rows(), Hff.cols()) * lambda + Hff);
    // Hff_inv_llt.solveInPlace(Hff_inv);
    // TODO: Use Cholesky, however, often unstable without adapting lambda.

    data.Hff_inv.topLeftCorner(data.free_idx.size(), data.free_idx.size()) =
        (Eigen::MatrixXd::Identity(data.free_idx.size(), data.free_idx.size()) * lambda +
         data.Hff.topLeftCorner(data.free_idx.size(), data.free_idx.size()))
            .inverse();

    // data.Hff_inv_llt.compute(data.I * lambda + H);
    // data.Hff_inv_llt.solveInPlace(data.Hff_inv);

    if (data.clamped_idx.size() == 0) {
      data.delta_xf = -data.Hff_inv * (data.q_free) - data.x_free;
    } else {
      data.delta_xf.head(data.free_idx.size()) =
          -data.Hff_inv.topLeftCorner(data.free_idx.size(), data.free_idx.size()) *
              (data.q_free.head(data.free_idx.size()) +
               data.Hfc.topLeftCorner(data.free_idx.size(), data.clamped_idx.size()) *
                   data.x_clamped.head(data.clamped_idx.size())) -
          data.x_free.head(data.free_idx.size());
    }

    data.f_old = (0.5 * data.x.transpose() * H * data.x + q.transpose() * data.x)(0);
    static const Eigen::VectorXd alpha_space = Eigen::VectorXd::LinSpaced(10, 1.0, 0.1);

    data.armijo_reached = false;
    for (int ai = 0; ai < alpha_space.rows(); ++ai) {
      data.x_new = data.x;
      for (size_t i = 0; i < data.free_idx.size(); ++i) {
        data.x_new(data.free_idx[i]) =
            std::max(std::min(data.x(data.free_idx[i]) + alpha_space[ai] * data.delta_xf(i), b_high(i)), b_low(i));
      }

      data.f_new = (0.5 * data.x_new.transpose() * H * data.x_new + q.transpose() * data.x_new)(0);
      data.x_diff.noalias() = data.x - data.x_new;

      // armijo criterion
      const double armijo_coef =
          (data.f_old - data.f_new) / (data.grad.transpose() * data.x_diff + 1e-5);  // TODO: Check 1e-5
      if (armijo_coef > gamma) {
        data.armijo_reached = true;
        data.x = data.x_new;
        break;
      }
    }

    // break if no step made
    if (!data.armijo_reached) break;
  }

  return;  // {data.Hff_inv, data.x, data.free_idx, data.clamped_idx, data.it};
}
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
