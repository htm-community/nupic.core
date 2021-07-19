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
 * Implement the CoordinateEncoder
 */

#include <htm/encoders/CoordinateEncoder.hpp>

#include <cmath> // tgamma, M_PI
#include <murmurhash3/MurmurHash3.hpp>

using namespace std;
using namespace htm;

CoordinateEncoder::CoordinateEncoder(const CoordinateEncoderParameters &parameters)
  { initialize( parameters ); }

void CoordinateEncoder::initialize( const CoordinateEncoderParameters &parameters ) {
  // Check parameters.
  NTA_CHECK( parameters.size > 0u );
  NTA_CHECK( parameters.numDimensions > 0u );
  UInt num_active_args = 0;
  if( parameters.activeBits > 0u)   { num_active_args++; }
  if( parameters.sparsity   > 0.0f) { num_active_args++; }
  NTA_CHECK( num_active_args != 0u )
      << "Missing argument, need one of: 'activeBits' or 'sparsity'.";
  NTA_CHECK( num_active_args == 1u )
      << "Too many arguments, choose only one of: 'activeBits' or 'sparsity'.";

  // Save the parameters and fill in any missing ones.
  args_ = parameters;
  // Determine number of activeBits.
  if( args_.sparsity > 0.0f ) {
    NTA_CHECK( args_.sparsity >= 0.0f );
    NTA_CHECK( args_.sparsity <= 1.0f );
    args_.activeBits = (UInt) round( args_.size * args_.sparsity );
    NTA_CHECK( args_.activeBits > 0u );
  }
  // Determine sparsity. Always calculate this even if it was given, to correct for rounding error.
  args_.sparsity = (Real) args_.activeBits / args_.size;
  while( args_.seed == 0u ) {
    args_.seed = Random().getUInt32();
  }

  BaseEncoder<const vector<Real64> &>::initialize({ parameters.size });

  // Find radius of sphere in numDimensions & volume of activeBits.
  const Real volume     = (Real) args_.activeBits;
  const Real nd         = (Real) args_.numDimensions;
  const Real radiusToND = volume * tgamma( nd / 2.0f + 1.0f ) / pow( M_PI, nd / 2.0f );
  const Real radius     = pow( radiusToND, 1.0f / nd );
  args_.radius = (Real64) 2.0f * radius;

  // Find all coordinates inside of the sphere's bounding box.
  UInt maxExtent = (UInt) 2 * radius + 3; // Pad the box so that the sphere does not touch the edge.
  // Use an SDR to manage coordinates & conversions.
  vector<UInt> neighborhoodDimensions( args_.numDimensions, maxExtent );
  neighborhood_.initialize( neighborhoodDimensions );
  // Activate every location in the neighborhood.
  auto &dense = neighborhood_.getDense();
  fill( dense.begin(), dense.end(), 1u );
  neighborhood_.setDense( dense );
  const auto &coordinates = neighborhood_.getCoordinates();

  // Find how far each coordinate is from the center of the sphere.
  Real center = radius + 1.33f; // Center the sphere in the box, don't let it touch the boundaries.
  vector<Real> distances( neighborhood_.getSum() );
  for(UInt idx = 0; idx < neighborhood_.getSum(); ++idx) {
    Real d = 0.0f;
    for(UInt dim = 0; dim < args_.numDimensions; ++dim) {
      d += pow((Real) coordinates[dim][idx] - center, 2.0f);
    }
    distances[idx] = d;
  }

  // Sort the coordinates by distance from center of the sphere.
  const auto cmp = [&distances] (const UInt &A, const UInt &B)
                                        { return distances[A] < distances[B]; };
  auto &index = neighborhood_.getSparse();
  sort( index.begin(), index.end(), cmp );
  // Discard extra coordinates, from the corners of the bounding box.
  index.resize( args_.activeBits );
  // Save the coordinates of the bits we want.
  neighborhood_.setSparse( index );
}


void CoordinateEncoder::encode(const vector<Real64> &coordinates, SDR &output) {
  NTA_CHECK( coordinates.size() == args_.numDimensions );
  NTA_CHECK( output.size == args_.size );

  for(const auto &input : coordinates) {
    if( isnan(input) ) {
      output.zero();
      return;
    }
  }

  auto &data = output.getDense();
  fill( data.begin(), data.end(), 0u );

  // Find where we are in the space.
  vector<UInt> location( args_.numDimensions );
  for(UInt dim = 0; dim < args_.numDimensions; ++dim) {
    location[dim] = (UInt) (coordinates[dim] / args_.resolution);
  }

  // Iterate through the area near this location.  Hash each nearby location and
  // use each hash to set a bit in the output SDR.

  const auto &neigh = neighborhood_.getCoordinates();
  vector<UInt> hash_buffer( args_.numDimensions );
  for(UInt idx = 0; idx < neighborhood_.getSum(); ++idx) {
    for(UInt dim = 0; dim < args_.numDimensions; ++dim) {
      hash_buffer[dim] = location[dim] + neigh[dim][idx];
    }
    UInt32 bucket = MurmurHash3_x86_32( hash_buffer.data(),
                                        hash_buffer.size() * sizeof(UInt),
                                        args_.seed);
    data[bucket % output.size] = 1u;
  }
  output.setDense( data );
}
