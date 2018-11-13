# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# -----------------------------------------------------------------------------

# Combine multiple Unix static libraries into a single static library.
#
# This script is intended to be invoked via `${CMAKE_COMMAND} -DLIB_TARGET= ...`.
#
# ARGS:
#
#   LIB_TARGET: target name of resulting static library (passed to add_library)
#   TARGET_LOCATION: Full path to the target library
#   SRC_LIB_LOCATIONS: List of source static library paths
#   LIST_SEPARATOR: separator string that separates paths in
#     SRC_LIB_LOCATIONS; NOTE with cmake 2.8.11+, caller could use the generator
#     "$<SEMICOLON>" in SRC_LIB_LOCATIONS, and this arg would be unnecessary.
#   BINARY_DIR: The value of ${CMAKE_CURRENT_BINARY_DIR} from caller
#   CMAKE_AR: The value of ${CMAKE_AR} from caller

function(COMBINE_UNIX_ARCHIVES
         LIB_TARGET
         TARGET_LOCATION
         SRC_LIB_LOCATIONS
         LIST_SEPARATOR
         BINARY_DIR
         CMAKE_AR)

  message(STATUS
          "COMBINE_UNIX_ARCHIVES("
          "  LIB_TARGET=${LIB_TARGET}, "
          "  TARGET_LOCATION=${TARGET_LOCATION}, "
          "  SRC_LIB_LOCATIONS=${SRC_LIB_LOCATIONS}, "
          "  LIST_SEPARATOR=${LIST_SEPARATOR}, "
          "  BINARY_DIR=${BINARY_DIR}, "
          "  CMAKE_AR=${CMAKE_AR})")

  string(REPLACE ${LIST_SEPARATOR} ";"
         SRC_LIB_LOCATIONS "${SRC_LIB_LOCATIONS}")

  set(scratch_dir ${BINARY_DIR}/combine_unix_archives_${LIB_TARGET})
  file(MAKE_DIRECTORY ${scratch_dir})

  # Extract archives into individual directories to avoid object file collisions
  set(all_object_locations)
  foreach(lib ${SRC_LIB_LOCATIONS})
    message(STATUS "COMBINE_UNIX_ARCHIVES: LIB_TARGET=${LIB_TARGET}, src-lib=${lib}")
    # Create working directory for current source lib
    get_filename_component(basename ${lib} NAME)
    set(working_dir ${scratch_dir}/${basename}.dir)
    file(MAKE_DIRECTORY ${working_dir})

    # Extract objects from current source lib
    execute_process(COMMAND ${CMAKE_AR} -x ${lib}
                    WORKING_DIRECTORY ${working_dir}
                    RESULT_VARIABLE exe_result)
    if(NOT "${exe_result}" STREQUAL "0")
      message(FATAL_ERROR "COMBINE_UNIX_ARCHIVES: obj extraction process failed exe_result='${exe_result}'")
    endif()

    # Accumulate objects
    if(MSVC)
      # Windows; all COFF format objects
      set(globbing "${working_dir}/*.obj")
      message(STATUS "this is MSVC")
    elseif(MINGW)
      # Note: MinGW can generate both .o and .obj in same project.
      #       they must be converted to COFF/PE format objects.
      set(globbing "${working_dir}/*.o*")
      message(STATUS "this is MINGW")
    else()
      # Linux or OS X; all ELF format objects
      set(globbing "${working_dir}/*.o")
      message(STATUS "this is Linux like or OSx")
    endif()

    file(GLOB_RECURSE objects "${globbing}")
    if (NOT objects)
      file(GLOB_RECURSE working_dir_listing "${working_dir}/*")
      message(FATAL_ERROR
              "COMBINE_UNIX_ARCHIVES: no extracted object files from ${lib} "
              "found using globbing=${globbing}, "
              "but the following entries were found: ${working_dir_listing}.")
    endif()

    # Prepend source lib name to object name to help prevent collisions during
    # subsequent archive extractions. This helps guard against obj file
    # overwrites during object extraction in cases where the same object name
    # existed in two different source archives, but does not prevent the issue
    # if same-named objects exist in one archive.
    foreach(old_obj_file_path ${objects})
      get_filename_component(old_obj_file_name ${old_obj_file_path} NAME)
      set(new_obj_file_path "${working_dir}/${basename}-${old_obj_file_name}")
      file(RENAME ${old_obj_file_path} ${new_obj_file_path})
      # Use relative paths to work around too-long command failure when building
      # on Windows in AppVeyor
      file(RELATIVE_PATH new_obj_file_path ${scratch_dir} ${new_obj_file_path})
      
      ####### remove this section if we no longer support MINGW
      get_filename_component(ext ${new_obj_file_path} EXT)
      if (MINGW AND ${ext} STREQUAL ".o")
        # This is a hack to get around a problem of some objects being generated
	# as ELF format .o files when they need to be .obj COFF/PE format
	get_filename_component(fn ${new_obj_file_path} NAME_WE) 
	set(conv_prog "${REPOSITORY_DIR}/external/windows64-gcc/bin/objconv.exe")
	execute_process(COMMAND ${conv_prog} -fCOFF64 ${new_obj_file_path} ${fn}.obj
	    WORKING_DIRECTORY ${scratch_dir} RESULT_VARIABLE exe_result)
        if(NOT "${exe_result}" STREQUAL "0")
             message(FATAL_ERROR "COMBINE_UNIX_ARCHIVES: Cannot convert .o to .obj; exe_result='${exe_result}'")
        endif()
        set(new_obj_file_path ${fn}.obj)
      endif()
      ##########
      
      list(APPEND all_object_locations ${new_obj_file_path})
    endforeach()
  endforeach()

  # Generate the target static library from all source objects
  file(TO_NATIVE_PATH ${TARGET_LOCATION} TARGET_LOCATION)
  execute_process(COMMAND ${CMAKE_AR} rcs ${TARGET_LOCATION} ${all_object_locations}
                  WORKING_DIRECTORY ${scratch_dir} RESULT_VARIABLE exe_result)
  if(NOT "${exe_result}" STREQUAL "0")
    message(FATAL_ERROR "COMBINE_UNIX_ARCHIVES: archive construction process failed exe_result='${exe_result}'")
  endif()

  # Remove scratch directory
  file(REMOVE_RECURSE ${scratch_dir})
endfunction(COMBINE_UNIX_ARCHIVES)


combine_unix_archives(${LIB_TARGET}
                      ${TARGET_LOCATION}
                      "${SRC_LIB_LOCATIONS}"
                      ${LIST_SEPARATOR}
                      ${BINARY_DIR}
                      ${CMAKE_AR})
