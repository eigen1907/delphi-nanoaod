# Modern DELPHI Analysis Examples
#
# FindDELPHI.cmake
#
# Some common utility functions
#
# Dietrich Liko <Dietrich.Liko@oeaw.ac.at>

cmake_minimum_required(VERSION 3.26)

project(DELPHI VERSION 0.1 LANGUAGES Fortran CXX)

# skelana.cra.in resolves (DELPHI_PAM)/phdstxx.car etc. nypatchy needs DELPHI_PAM at
# build time. When CMake/Make runs without sourcing delphi setup, set a default.
if(NOT DEFINED DELPHI_PAM_DIRECTORY OR DELPHI_PAM_DIRECTORY STREQUAL "")
    if(NOT "$ENV{DELPHI_PAM}" STREQUAL "")
        set(DELPHI_PAM_DIRECTORY "$ENV{DELPHI_PAM}" CACHE PATH "Directory containing DELPHI .car sources for nypatchy (PHDSTXX, skelana, …)")
    else()
        foreach(_d
                "/cvmfs/delphi.cern.ch/releases/rhel-9-x86_64/latest/dstana/161018/src/car"
                "/cvmfs/delphi.cern.ch/releases/almalinux-9-x86_64/latest/dstana/161018/src/car")
            if(EXISTS "${_d}/phdstxx.car")
                set(DELPHI_PAM_DIRECTORY "${_d}" CACHE PATH "Directory containing DELPHI .car sources for nypatchy (PHDSTXX, skelana, …)")
                break()
            endif()
        endforeach()
        if(NOT DEFINED DELPHI_PAM_DIRECTORY OR DELPHI_PAM_DIRECTORY STREQUAL "")
            message(WARNING "DELPHI_PAM_DIRECTORY not set and no default phdstxx.car found; nypatchy will fail unless you export DELPHI_PAM or pass -DDELPHI_PAM_DIRECTORY=... (after: source /cvmfs/delphi.cern.ch/setup.sh)")
        endif()
    endif()
endif()

# Wrapper for nypatchy, the DELPHI source code manager
# 
# Dependencies to CAR files can be added to the command
#
# ypatchy(<Fortran File> <CRA File> [<CAR Files> ...)
#
# Example
#
#    configure_file(example01.cra.in example01.cra)
#    ypatchy(example01.f example01.cra)
#
function(ypatchy FOUT CRA)

    if(DELPHI_PAM_DIRECTORY STREQUAL "")
        message(FATAL_ERROR "DELPHI_PAM_DIRECTORY is empty; source /cvmfs/delphi.cern.ch/setup.sh before cmake, or pass -DDELPHI_PAM_DIRECTORY=/path/to/dstana/.../src/car")
    endif()

    add_custom_command(
        OUTPUT ${FOUT}
        MAIN_DEPENDENCY ${CRA}
        DEPENDS ${ARGN}
        COMMAND ${CMAKE_COMMAND} -E env "DELPHI_PAM=${DELPHI_PAM_DIRECTORY}"
            nypatchy - ${FOUT} ${CRA} nypatchy.ylog .go
        COMMENT "Running nypatchy on ${CRA}"
        VERBATIM
        )

endfunction()

# Wrapper for the cernlib command that simplifies link options
function(target_link_cernlibs TARGET)

    execute_process(
        COMMAND cernlib ${ARGN}
        OUTPUT_VARIABLE CERNLIBS_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    string(REPLACE "-lnsl" "" CERNLIBS "${CERNLIBS_RAW}")
    target_link_libraries(${TARGET} ${CERNLIBS})

endfunction()

# Wrapper for the dellib command that simplifies link options
function(target_link_dellibs TARGET)

    execute_process(
        COMMAND dellib ${ARGN}
        OUTPUT_VARIABLE DELLIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    target_link_libraries(${TARGET} ${DELLIBS})

endfunction()