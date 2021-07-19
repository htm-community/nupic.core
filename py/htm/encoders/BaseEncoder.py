# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2020, David Keeney
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------


# Base class for all encoders.
# An encoder converts a value to a sparse distributed representation.
#
# Subclasses must implement method encode and Serializable interface.
# Subclasses can optionally implement method reset.
#
# There are several critical properties which all encoders must have:
#
# 1) Semantic similarity:  Similar inputs should have high overlap.  Overlap
# decreases smoothly as inputs become less similar.  Dissimilar inputs have
# very low overlap so that the output representations are not easily confused.
#
# 2) Stability:  The representation for an input does not change during the
# lifetime of the encoder.
#
# 3) Sparsity: The output SDR should have a similar sparsity for all inputs and
# have enough active bits to handle noise and subsampling.
#
# Reference: https://arxiv.org/pdf/1602.05925.pdf


# Members dimensions & size describe the shape of the encoded output SDR.
# For example, a 6 by 4 dimension would be (6,4,)
# size is the total number of bits in the result.
# A subclass of the BaseEncoder should be passed the dimension in the constructor.
# All subclasses must contain an encode( ) method and a reset( ) method.

from abc import ABC, abstractmethod
import numpy as np
from htm.bindings.sdr import SDR

class BaseEncoder(ABC):

  @abstractmethod
  def __init__(dimensions):
    self.dimensions = dimensions
    self.size = SDR(dimensions).size
    
  @abstractmethod
  def reset(self):
    raise NotImplementedError()
    
  @abstractmethod
  def encode(self, input, output):
    raise NotImplementedError()