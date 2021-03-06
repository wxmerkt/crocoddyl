///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/actuations/multicopter-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationModelMultiCopterBase() {
  bp::class_<ActuationModelMultiCopterBase, bp::bases<ActuationModelAbstract> >(
      "ActuationModelMultiCopterBase",
      "Actuation models with base actuated by several propellers (e.g. aerial manipulators).",
      bp::init<boost::shared_ptr<StateMultibody>, int, Eigen::MatrixXd>(
          bp::args("self", "state", "nrotors", "force_torque"),
          "Initialize the full actuation model.\n\n"
          ":param state: state of multibody system, \n"
          ":param nrotors: number of rotors of the flying base, \n"
          ":param force_torque: matrix that maps rotors thrust to generalized torque of the flying base."))
      .def("calc", &ActuationModelMultiCopterBase::calc, bp::args("self", "data", "x", "u"),
           "Compute the actuation signal from the control input u.\n\n"
           ":param data: multicopter-base actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", &ActuationModelMultiCopterBase::calcDiff, bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the full actuation. It assumes that you\n"
           "create the data using this class. The reason is that the derivatives are constant and\n"
           "defined in createData. The Hessian is constant, so we don't write again this value.\n"
           ":param data: multicopter-base actuation data\n"
           ":param x: state vector\n"
           ":param u: control input\n")
      .def("createData", &ActuationModelMultiCopterBase::createData, bp::args("self"),
           "Create the multicopter-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("nrotors",
                    bp::make_function(&ActuationModelMultiCopterBase::get_nrotors,
                                      bp::return_value_policy<bp::copy_const_reference>()),
                    "Number of rotors in the flying base")
      .add_property(
          "tauf",
          bp::make_function(&ActuationModelMultiCopterBase::get_tauf, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ActuationModelMultiCopterBase::set_tauf), "Matrix mapping from thrusts to body torque");
}

}  // namespace python
}  // namespace crocoddyl
