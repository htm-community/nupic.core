/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
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
 * ----------------------------------------------------------------------
 */

/** @file
 * Definitions for the base Serializable class in C++
 *
 * CapnProto serialization has been removed and replaced with binary streams.
 *    dkeeney 8/15/2018
 */

#ifndef NTA_SERIALIZABLE_HPP
#define NTA_SERIALIZABLE_HPP


#include <iostream>
#include <fstream>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/ImportFilesystem.hpp>

#define CEREAL_SAVE_FUNCTION_NAME save_ar
#define CEREAL_LOAD_FUNCTION_NAME load_ar
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

// The RapidJson distribution (Used by Cereal) had a problem with these warnings.
// It is being fixed in next release.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexceptions"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

#include <cereal/archives/json.hpp>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#define SERIALIZABLE_VERSION 3


namespace nupic {

/**
 * Base Serializable class that any serializable class
 * should inherit from.
 */
class Serializable {
public:
  Serializable() {}
  virtual inline int getSerializableVersion() const { return SERIALIZABLE_VERSION; }

  virtual inline void saveToFile(std::string filePath) const {
    std::string dirPath = Path::getParent(filePath);
	  Directory::create(dirPath, true, true);
	  std::ofstream out(filePath, std::ios_base::out | std::ios_base::binary);
	  out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	  out.precision(std::numeric_limits<double>::digits10 + 1);
	  out.precision(std::numeric_limits<float>::digits10 + 1);
	  save(out);
		//cereal::BinaryOutputArchive archive( out );
		//save_ar(archive);
	  out.close();
  }

  virtual inline void loadFromFile(std::string filePath) {
    std::ifstream in(filePath, std::ios_base::in | std::ios_base::binary);
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    load(in);
		//cereal::BinaryInputArchive archive( in );
		//load_ar(archive);
    in.close();
  }

  // These must be implemented by the subclass.
  virtual void save(std::ostream &stream) const = 0;
  virtual void load(std::istream &stream) = 0;

	//template <class Archive>
	//void save_ar(Archive& ar) const { };
	//template <class Archive>
	//void load_ar(Archive& ar) { };


  virtual ~Serializable() {}
};

} // end namespace nupic

#endif // NTA_SERIALIZABLE_HPP

