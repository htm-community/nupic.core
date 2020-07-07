/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Implementation for DatabaseOutRegion class
 */

#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <stdio.h>

#include <htm/engine/Output.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/regions/DatabaseOutRegion.hpp>
#include <htm/utils/Log.hpp>

#define MAX_NUMBER_OF_INPUTS 10 // maximal number of inputs/scalar streams in the database

namespace htm {

DatabaseOutRegion::DatabaseOutRegion(const ValueMap &params, Region* region)
    : RegionImpl(region), filename_(""),
	  dbHandle(nullptr) {
  if (params.contains("outputFile")) {
    std::string s = params.getString("outputFile", "");
    openFile(s);
  }
  else
    filename_ = "";

}

DatabaseOutRegion::DatabaseOutRegion(ArWrapper& wrapper, Region* region)
    : RegionImpl(region), filename_(""),
	  dbHandle(nullptr) {
  cereal_adapter_load(wrapper);
}


DatabaseOutRegion::~DatabaseOutRegion() { closeFile(); }

void DatabaseOutRegion::initialize() {
  NTA_CHECK(region_ != nullptr);
  // We have no outputs or parameters; just need our input.
  const std::map<std::string, std::shared_ptr<Input>> inputs = region_->getInputs();


  NTA_ASSERT(inputs.size()!=0) << "DatabaseOutRegion::initialize - no inputs configured\n";

	for (const auto & inp : inputs) {
		const auto inObj = inp.second;
		if (inObj->hasIncomingLinks() && inObj->getData().getCount() != 0) { //create tables only for those, whose was configured
			createTable("dataStream_" + inp.first);
		}
	}

}

void DatabaseOutRegion::createTable(const std::string &sTableName){

	/* Create SQL statement */
	std::string  sql = "CREATE TABLE "+sTableName+" (iteration INTEGER PRIMARY KEY, value);";

	char *zErrMsg;
	/* Execute SQL statement */
	int returnCode = sqlite3_exec(dbHandle, sql.c_str(), nullptr, 0, &zErrMsg);

	if( returnCode != SQLITE_OK ){
		NTA_THROW << "Error creating SQL table, message:"
				  << zErrMsg;
		sqlite3_free(zErrMsg);
	}
}

void DatabaseOutRegion::insertData(const std::string &sTableName, unsigned int iIteration, const std::shared_ptr<Input> inputData){

	NTA_ASSERT(inputData->getData().getCount()==1);

	Real32* value = (Real32 *)inputData->getData().getBuffer();
	/* Create SQL statement */
	std::string  sql = "INSERT INTO "+sTableName+" VALUES ("+std::to_string(iIteration)+","+ std::to_string(*value) + ");";

	char *zErrMsg;

	int returnCode = sqlite3_exec(dbHandle, sql.c_str(), nullptr, 0, &zErrMsg);

	if( returnCode != SQLITE_OK ){
		NTA_THROW << "Error inserting data to SQL table, message:"
				  << zErrMsg;
		sqlite3_free(zErrMsg);
	}
}

void DatabaseOutRegion::compute() {


	const std::map<std::string, std::shared_ptr<Input>> inputs = region_->getInputs();


	NTA_ASSERT(inputs.size()!=0) << "DatabaseOutRegion::initialize - no inputs configured\n";

	for (const auto & inp : inputs)
	{
		const auto inObj = inp.second;
		if (inObj->hasIncomingLinks() && inObj->getData().getCount() != 0) { //create tables only for those, whose was configured
			insertData("dataStream_" + inp.first, iIterationCounter, inObj);
		}
	}

	iIterationCounter += 1; // increase inner counter of iteration by one
}

void DatabaseOutRegion::closeFile() {
  if (dbHandle!=NULL) {
	sqlite3_close(dbHandle);
	dbHandle = nullptr;
    filename_ = "";
  }
}

void DatabaseOutRegion::openFile(const std::string &filename) {

  if (dbHandle != NULL)
    closeFile();
  if (filename == "")
    return;

  // if the database file exists delete it
	std::ifstream ifile;
	ifile.open(filename.c_str());
	if(ifile) {
		//file exits so delete it
		ifile.close();
		if(remove(filename.c_str())!=0)
			NTA_THROW << "DatabaseOutRegion::openFile -- Error deleting existing database file! Filename:"
					  << filename;

	} else {
		ifile.close();
	}

  // create new file

  int result = sqlite3_open(filename.c_str(), &dbHandle);

  if (result!=0 || dbHandle==nullptr)
  {

    NTA_THROW
        << "DatabaseOutRegion::openFile -- unable to create database file: "
        << filename.c_str()
		<< " Error code:"
		<< result;
  }
  filename_ = filename;

  iIterationCounter = 0; // set iteration counter to 0

}

void DatabaseOutRegion::setParameterString(const std::string &paramName,
                                            Int64 index, const std::string &s) {

  if (paramName == "outputFile") {
    if (s == filename_)
      return; // already set
    if (dbHandle!=nullptr)
      closeFile();
    openFile(s);
  } else {
    NTA_THROW << "DatabaseOutRegion -- Unknown string parameter " << paramName;
  }
}

std::string DatabaseOutRegion::getParameterString(const std::string &paramName,
                                                   Int64 index) {
  if (paramName == "outputFile") {
    return filename_;
  } else {
    NTA_THROW << "DatabaseOutRegion -- unknown parameter " << paramName;
  }
}

std::string
DatabaseOutRegion::executeCommand(const std::vector<std::string> &args,
                                   Int64 index) {
  NTA_CHECK(args.size() > 0);
  // Process the flushFile command
  if (args[0] == "closeFile") {
    closeFile();
  } else {
    NTA_THROW << "DatabaseOutRegion: Unknown execute '" << args[0] << "'";
  }

  return "";
}

Spec *DatabaseOutRegion::createSpec() {

  auto ns = new Spec;
  ns->description =
      "DatabaseOutRegion is a node that writes multiple scalar streams "
      "to a SQLite3 database file (.db). The target filename is specified "
      "using the 'outputFile' parameter at run time. On each "
      "compute, all inputs are written "
      "to the database.\n";

  for (int i = 0; i< MAX_NUMBER_OF_INPUTS; i ++){ // create 10 inputs, user don't have to use them all
		ns->inputs.add("dataIn"+std::to_string(i),
								InputSpec("Data scalar to be written to the database",
													NTA_BasicType_Real32,
													0,     // count
													false, // required?
													true, // isRegionLevel
													true   // isDefaultInput
													));
  }

  ns->parameters.add("outputFile",
              ParameterSpec("Writes data stream to this database file on each "
                            "compute. Database is recreated on initialization "
                            "This parameter must be set at runtime before "
                            "the first compute is called. Throws an "
                            "exception if it is not set or "
                            "the file cannot be written to.\n",
                            NTA_BasicType_Byte,
                            0,  // elementCount
                            "", // constraints
                            "", // defaultValue
                            ParameterSpec::ReadWriteAccess));

  ns->commands.add("closeFile",
                   CommandSpec("Close the current database file, if open."));

  return ns;
}

size_t DatabaseOutRegion::getNodeOutputElementCount(const std::string &outputName) const {
  NTA_THROW
      << "DatabaseOutRegion::getNodeOutputElementCount -- unknown output '"
      << outputName << "'";
}


bool DatabaseOutRegion::operator==(const RegionImpl &o) const {
  if (o.getType() != "DatabaseOutRegion") return false;
  DatabaseOutRegion& other = (DatabaseOutRegion&)o;
  if (filename_ != other.filename_) return false;

  return true;
}

} // namespace htm
