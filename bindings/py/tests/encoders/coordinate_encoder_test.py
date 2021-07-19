# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, David McDougall
#
# The following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

"""Unit tests for Coordinate Encoder."""

import pickle
import numpy as np
import unittest

from htm.bindings.sdr import SDR, Metrics
from htm.bindings.encoders import CoordinateEncoder, CoordinateEncoderParameters

class CoordinateEncoderTest(unittest.TestCase):
    def testConstructor(self):
        params1 = CoordinateEncoderParameters()
        params1.numDimensions = 1
        params1.size     = 100
        params1.sparsity = .10
        params1.radius   = 10
        R1 = CoordinateEncoder( params1 )

        params2 = R1.parameters
        params2.numDimensions = 1
        params2.size     = 100
        params2.sparsity = .10
        params2.radius   = 10
        R2 = CoordinateEncoder( params2 )

        A = SDR( R1.parameters.size )
        R1.encode( [66], A )

        B = R2.encode( [66] )
        assert( A == B )

    def testErrorChecks(self):
        params1 = CoordinateEncoderParameters()
        params1.size     = 100
        params1.sparsity = .10
        params1.radius   = 10
        # Test missing numDimensions
        with self.assertRaises(RuntimeError):
            R1 = CoordinateEncoder( params1 )

        params1.numDimensions = 1
        R1 = CoordinateEncoder( params1 )
        A = SDR([10, 10])
        R1.encode( [33], A )

        # Test wrong input dimensions
        B = SDR( 1 )
        with self.assertRaises(RuntimeError):
            R1.encode( [3], B )
        with self.assertRaises(RuntimeError):
            R1.encode( 3, A )

        # Test invalid parameters, size == 0
        params1.size = 0
        with self.assertRaises(RuntimeError):
            CoordinateEncoder( params1 )
        params1.size = 100

        # Test invalid parameters, activeBits == 0
        params1.activeBits = 0
        params1.sparsity = 0.00001 # Rounds to zero!
        with self.assertRaises(RuntimeError):
            CoordinateEncoder( params1 )

        # Test missing activeBits
        params2 = CoordinateEncoderParameters()
        params2.numDimensions = 1
        params2.size     = 100
        params2.radius   = 10
        with self.assertRaises(RuntimeError):
            CoordinateEncoder( params2 )
        # Test missing resolution/radius
        params3 = CoordinateEncoderParameters()
        params3.numDimensions = 1
        params3.size       = 100
        params3.activeBits = 10
        with self.assertRaises(RuntimeError):
            CoordinateEncoder( params3 )

        # Test too many parameters: activeBits & sparsity
        params4 = CoordinateEncoderParameters()
        params4.numDimensions = 1
        params4.size       = 100
        params4.sparsity   = .6
        params4.activeBits = 10
        params4.radius     = 4
        with self.assertRaises(RuntimeError):
            CoordinateEncoder( params4 )
        # Test too many parameters: resolution & radius
        params5 = CoordinateEncoderParameters()
        params5.numDimensions = 1
        params5.size       = 100
        params5.activeBits = 10
        params5.radius     = 4
        params5.resolution = 4
        with self.assertRaises(RuntimeError):
            CoordinateEncoder( params5 )

    def testSparsityActiveBits(self):
        """ Check that these arguments are equivalent. """
        # Round sparsity up
        P = CoordinateEncoderParameters()
        P.size     = 100
        P.sparsity = .0251
        P.radius   = 10
        R = CoordinateEncoder( P )
        assert( R.parameters.activeBits == 3 )
        # Round sparsity down
        P = CoordinateEncoderParameters()
        P.size     = 100
        P.sparsity = .0349
        P.radius   = 10
        R = CoordinateEncoder( P )
        assert( R.parameters.activeBits == 3 )
        # Check activeBits
        P = CoordinateEncoderParameters()
        P.size       = 100
        P.activeBits = 50 # No floating point issues here.
        P.radius     = 10
        R = CoordinateEncoder( P )
        assert( R.parameters.sparsity == .5 )

    def testRadiusResolution(self):
        """ Check that these arguments are equivalent. """
        # radius -> resolution
        P = CoordinateEncoderParameters()
        P.size       = 2000
        P.activeBits = 100
        P.radius     = 12
        R = CoordinateEncoder( P )
        self.assertAlmostEqual( R.parameters.resolution, 12./100, places=5)

        # resolution -> radius
        P = CoordinateEncoderParameters()
        P.size       = 2000
        P.activeBits = 100
        P.resolution = 12
        R = CoordinateEncoder( P )
        self.assertAlmostEqual( R.parameters.radius, 12*100, places=5)

        # Moving 1 resolution moves 1 bit (usually)
        P = CoordinateEncoderParameters()
        P.size       = 2000
        P.activeBits = 100
        P.resolution = 3.33
        P.seed       = 42
        R = CoordinateEncoder( P )
        sdrs = []
        for i in range(100):
            X = i * ( R.parameters.resolution )
            sdrs.append( R.encode( X ) )
            print("X", X, sdrs[-1])
        moved_one = 0
        moved_one_samples = 0
        for A, B in zip(sdrs, sdrs[1:]):
            delta = A.getSum() - A.getOverlap( B )
            if A.getSum() == B.getSum():
                assert( delta < 2 )
                moved_one += delta
                moved_one_samples += 1
        assert( moved_one >= .9 * moved_one_samples )

    def testAverageOverlap(self):
        """ Verify that nearby values have the correct amount of semantic
        similarity. Also measure sparsity & activation frequency. """
        P = CoordinateEncoderParameters()
        P.size     = 2000
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = CoordinateEncoder( P )
        A = SDR( R.parameters.size )
        num_samples = 10000
        M = Metrics( A, num_samples + 1 )
        for i in range( num_samples ):
            R.encode( i, A )
        print( M )
        assert(M.overlap.min()  > (1 - 1. / R.parameters.radius) - .04 )
        assert(M.overlap.max()  < (1 - 1. / R.parameters.radius) + .04 )
        assert(M.overlap.mean() > (1 - 1. / R.parameters.radius) - .001 )
        assert(M.overlap.mean() < (1 - 1. / R.parameters.radius) + .001 )
        assert(M.sparsity.min()  > R.parameters.sparsity - .01 )
        assert(M.sparsity.max()  < R.parameters.sparsity + .01 )
        assert(M.sparsity.mean() > R.parameters.sparsity - .005 )
        assert(M.sparsity.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.min()  > R.parameters.sparsity - .05 )
        assert(M.activationFrequency.max()  < R.parameters.sparsity + .05 )
        assert(M.activationFrequency.mean() > R.parameters.sparsity - .005 )
        assert(M.activationFrequency.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.entropy() > .99 )

    def testRandomOverlap(self):
        """ Verify that distant values have little to no semantic similarity.
        Also measure sparsity & activation frequency. """
        P = CoordinateEncoderParameters()
        P.numDimensions = 1
        P.size     = 2000
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = CoordinateEncoder( P )
        num_samples = 1000
        A = SDR( R.parameters.size )
        M = Metrics( A, num_samples + 1 )
        print(R.parameters.radius)
        for i in range( num_samples ):
            X = i * R.parameters.radius
            R.encode( [X], A )
        print( M )
        assert(M.overlap.max()  < .15 )
        assert(M.overlap.mean() < .10 )
        assert(M.sparsity.min()  > R.parameters.sparsity - .01 )
        assert(M.sparsity.max()  < R.parameters.sparsity + .01 )
        assert(M.sparsity.mean() > R.parameters.sparsity - .005 )
        assert(M.sparsity.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.min()  > R.parameters.sparsity - .05 )
        assert(M.activationFrequency.max()  < R.parameters.sparsity + .05 )
        assert(M.activationFrequency.mean() > R.parameters.sparsity - .005 )
        assert(M.activationFrequency.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.entropy() > .99 )

    def testDeterminism(self):
        """ Verify that the same seed always gets the same results. """
        GOLD = SDR( 1000 )
        GOLD.sparse = []

        P = CoordinateEncoderParameters()
        P.size     = GOLD.size
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = CoordinateEncoder( P )
        A = R.encode( [987654] )
        print( A )
        assert( A == GOLD )

    def testSeed(self):
        P = CoordinateEncoderParameters()
        P.numDimensions = 1
        P.size     = 1000
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 98
        R = CoordinateEncoder( P )
        A = R.encode( [987654] )

        P.seed = 99
        R = CoordinateEncoder( P )
        B = R.encode( [987654] )
        assert( A != B )

    def testPickle(self):
        """
        The pickling is successfull if pickle serializes and de-serialize the
        object, and both copies of the object yield identical encodings.
        """
        params = CoordinateEncoderParameters()
        params.numDimensions = 3
        params.size = 200
        params.sparsity = 0.05
        params.resolution = 0.1
        params.seed = 42
        enc = CoordinateEncoder(params)

        enc2 = pickle.loads(pickle.dumps(rdse, f))

        value_to_encode = (67, -3., 0)
        SDR_original = enc.encode(value_to_encode)
        SDR_loaded   = enc2.encode(value_to_encode)
        assert SDR_original == SDR_loaded
