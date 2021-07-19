/* ----------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
 * The following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <htm/encoders/CoordinateEncoder.hpp>

namespace py = pybind11;

using namespace std;
using namespace htm;

namespace htm_ext
{
  void init_CoordinateEncoder(py::module& m)
  {
    py::class_<CoordinateEncoderParameters>
                            py_CoordEncParams(m, "CoordinateEncoderParameters",
R"(Parameters for the CoordinateEncoder

Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
of them.)");

    py_CoordEncParams.def(py::init<>());

    py_CoordEncParams.def_readwrite("numDimensions", &CoordinateEncoderParameters::numDimensions,
R"(Member "numDimensions" is length of the input coordinate vector.)");

    py_CoordEncParams.def_readwrite("size", &CoordinateEncoderParameters::size,
R"(Member "size" is the total number of bits in the encoded output SDR.)");

    py_CoordEncParams.def_readwrite("sparsity", &CoordinateEncoderParameters::sparsity,
R"(Member "sparsity" is the fraction of bits in the encoded output which this
encoder will activate. This is an alternative way to specify the member
"activeBits".)");

    py_CoordEncParams.def_readwrite("activeBits", &CoordinateEncoderParameters::activeBits,
R"(Member "activeBits" is the number of true bits in the encoded output SDR.)");

    py_CoordEncParams.def_readwrite("radius", &CoordinateEncoderParameters::radius,
R"(Two inputs separated by more than the radius will have non-overlapping
representations. Two inputs separated by less than the radius will in general
overlap in at least some of their bits.)");

    py_CoordEncParams.def_readwrite("resolution", &CoordinateEncoderParameters::resolution,
R"(Two inputs separated by greater than, or equal to the resolution will
in general have different representations.)");

    py_CoordEncParams.def_readwrite("seed", &CoordinateEncoderParameters::seed,
R"(Member "seed" forces different encoders to produce different outputs, even if
the inputs and all other parameters are the same.  Two encoders with the same
seed, parameters, and input will produce identical outputs.

The seed 0 is special.  Seed 0 is replaced with a random number.)");

    py::class_<CoordinateEncoder> py_CoordinateEncoder(m, "CoordinateEncoder",
R"(TODO: DOCSTRING!)");

    py_CoordinateEncoder.def(py::init<CoordinateEncoderParameters&>());

    py_CoordinateEncoder.def_property_readonly("dimensions",
        [](const CoordinateEncoder &self) { return self.dimensions; });
    py_CoordinateEncoder.def_property_readonly("size",
        [](const CoordinateEncoder &self) { return self.size; });
    py_CoordinateEncoder.def_property_readonly("parameters",
        [](const CoordinateEncoder &self) { return self.parameters; },
R"(Contains the parameter structure which this encoder uses internally. All
fields are filled in automatically.)");

    py_CoordinateEncoder.def("encode", &CoordinateEncoder::encode);

    py_CoordinateEncoder.def("encode", []
        (CoordinateEncoder &self, vector<Real64> value) {
            auto sdr = new SDR({ self.size });
            self.encode( value, *sdr );
            return sdr;
    });

    py_CoordinateEncoder.def(py::pickle(
      [](const CoordinateEncoder& self) {
        std::stringstream ss;
        self.save(ss);
        return py::bytes( ss.str() );
      },
      [](py::bytes &s) {
        std::stringstream ss( s.cast<std::string>() );
        std::unique_ptr<CoordinateEncoder> self(new CoordinateEncoder());
        self->load(ss);
        return self;
    }));
  }
}
