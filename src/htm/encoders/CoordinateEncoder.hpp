/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
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
 * --------------------------------------------------------------------- */

/** @file
 * Define the CoordinateEncoder & CoordinateEncoderParameters
 */

#ifndef NTA_ENCODERS_COORD
#define NTA_ENCODERS_COORD

#include <htm/types/Sdr.hpp>
#include <htm/types/Types.hpp>
#include <htm/utils/Random.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/encoders/RandomDistributedScalarEncoder.hpp>

namespace htm {

/*
 * Parameters for the CoordinateEncoder
 *
 * Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
 * of them.
 */
struct CoordinateEncoderParameters {
  /*
   * Member "numDimensions" is length of the input coordinate vector.
   */
  UInt numDimensions = 0u;

  /**
   * Member "size" is the total number of bits in the encoded output SDR.
   */
  UInt size = 0u;

  /**
   * Member "activeBits" is the number of true bits in the encoded output SDR.
   */
  UInt activeBits = 0u;

  /**
   * Member "sparsity" is the fraction of bits in the encoded output which this
   * encoder will activate. This is an alternative way to specify the member
   * "activeBits".
   */
  Real sparsity = 0.0f;

  /**
   * Member "radius" Two inputs separated by more than the radius have
   * non-overlapping representations. Two inputs separated by less than the
   * radius will in general overlap in at least some of their bits. You can
   * think of this as the radius of the input.
   */
  Real radius = 0.0f;

  /**
   * Member "resolution" Two inputs separated by greater than, or equal to the
   * resolution will in general have different representations.
   */
  Real resolution = 1.0f;

  /**
   * Member "seed" forces different encoders to produce different outputs, even
   * if the inputs and all other parameters are the same.  Two encoders with the
   * same seed, parameters, and input will produce identical outputs.
   *
   * The seed 0 is special.  Seed 0 is replaced with a random number.
   */
  UInt seed = 0u;
};

/*
 * TODO: DOCUMENTATION
 TODO: EXPLAIN how this works...
 */
class CoordinateEncoder : public BaseEncoder<const std::vector<Real64> &>
{
public:
  CoordinateEncoder() {}
  CoordinateEncoder( const CoordinateEncoderParameters &parameters );
  void initialize( const CoordinateEncoderParameters &parameters );

  const CoordinateEncoderParameters &parameters = args_;

  void encode(const std::vector<Real64> &coordinates, SDR &output) override;

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    std::string name = "CoordinateEncoder";
    ar(cereal::make_nvp("name", name));
    ar(cereal::make_nvp("size", args_.size));
    ar(cereal::make_nvp("activeBits", args_.activeBits));
    ar(cereal::make_nvp("sparsity", args_.sparsity));
    ar(cereal::make_nvp("radius", args_.radius));
    ar(cereal::make_nvp("resolution", args_.resolution));
    ar(cereal::make_nvp("seed", args_.seed));
    ar(cereal::make_nvp("neighborhood", neighborhood_));
  }
  // FOR Cereal Deserialization
  template<class Archive>
  void load_ar(Archive& ar) {
    std::string name;
    ar(cereal::make_nvp("name", name));
    NTA_CHECK(name == "CoordinateEncoder");
    ar(cereal::make_nvp("size", args_.size));
    ar(cereal::make_nvp("activeBits", args_.activeBits));
    ar(cereal::make_nvp("sparsity", args_.sparsity));
    ar(cereal::make_nvp("radius", args_.radius));
    ar(cereal::make_nvp("resolution", args_.resolution));
    ar(cereal::make_nvp("seed", args_.seed));
    ar(cereal::make_nvp("neighborhood", neighborhood_));
    // BaseEncoder<Real64>::initialize({ args_.size }); // TODO!
  }

  ~CoordinateEncoder() override {};

private:
  CoordinateEncoderParameters args_;
  SDR neighborhood_;
};     // End class CoordinateEncoder
}      // End namespace
#endif // End ifdef NTA_ENCODERS_COORD
