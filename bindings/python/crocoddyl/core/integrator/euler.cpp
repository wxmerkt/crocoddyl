///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/integrator/euler.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionEuler() {
  bp::class_<IntegratedActionModelEuler, bp::bases<ActionModelAbstract> >(
      "IntegratedActionModelEuler",
      "Sympletic Euler integrator for differential action models.\n\n"
      "This class implements a sympletic Euler integrator (a.k.a semi-implicit\n"
      "integrator) give a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], [v + a * dt, a * dt] * dt).",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the sympletic Euler integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time\n"
          ":param withCostResidual: includes the cost residuals and derivatives."))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelEuler::calc, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete derivatives of a differential action model.\n\n"
          "It computes the time-discrete partial derivatives of a differential\n"
          "action model. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelEuler::createData, bp::args("self"),
           "Create the Euler integrator data.")
      .add_property("differential",
                    bp::make_function(&IntegratedActionModelEuler::get_differential,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &IntegratedActionModelEuler::set_differential, "differential action model")
      .add_property(
          "dt", bp::make_function(&IntegratedActionModelEuler::get_dt, bp::return_value_policy<bp::return_by_value>()),
          &IntegratedActionModelEuler::set_dt, "step time");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataEuler> >();

  bp::class_<IntegratedActionDataEuler, bp::bases<ActionDataAbstract> >(
      "IntegratedActionDataEuler", "Sympletic Euler integrator data.",
      bp::init<IntegratedActionModelEuler*>(bp::args("self", "model"),
                                            "Create sympletic Euler integrator data.\n\n"
                                            ":param model: sympletic Euler integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataEuler::differential, bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property("dx", bp::make_getter(&IntegratedActionDataEuler::dx, bp::return_internal_reference<>()),
                    "state rate.")
      .add_property("ddx_dx", bp::make_getter(&IntegratedActionDataEuler::ddx_dx, bp::return_internal_reference<>()),
                    "Jacobian of the state rate with respect to the state.")
      .add_property("ddx_du", bp::make_getter(&IntegratedActionDataEuler::ddx_du, bp::return_internal_reference<>()),
                    "Jacobian of the state rate with respect to the control.")
      .add_property("dxnext_dx",
                    bp::make_getter(&IntegratedActionDataEuler::dxnext_dx, bp::return_internal_reference<>()),
                    "Jacobian of the next state with respect to the state.")
      .add_property("dxnext_ddx",
                    bp::make_getter(&IntegratedActionDataEuler::dxnext_ddx, bp::return_internal_reference<>()),
                    "Jacobian of the next state with respect to the state rate.");
}

}  // namespace python
}  // namespace crocoddyl
