set(CMAKE_CXX_STANDARD 17)

# Unlike delphi-nanoaod, this target does NOT depend on SKELANA / nypatchy /
# fastjet. It links directly against PHDST and walks the ZEBRA store via
# the LQ/IQ/Q macros exposed in delphi-analysis/include/phdst/.
add_executable(delphi-raw-nanoaod
    src/delphi-raw-nanoaod.cpp
    src/raw_nanoaod_writer.cpp
)

add_executable(delphi-raw-bank-lister
    src/delphi-raw-bank-lister.cpp
)

target_compile_options(delphi-raw-nanoaod PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-depth=50>)   # argparse

target_include_directories(delphi-raw-nanoaod PRIVATE
    ${ROOT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(delphi-raw-nanoaod argparse delphi-analysis)
target_link_libraries(delphi-raw-nanoaod ROOT::Core ROOT::RIO ROOT::ROOTNTuple ROOT::Tree)
target_link_libraries(delphi-raw-bank-lister argparse delphi-analysis)

# Same DELPHI + CERNLIB link pattern as delphi-nanoaod, minus SKELANA itself.
# We still bring skelana in because phdst_analysis shares a translation unit
# with skelana_analysis in libdelphi-analysis.a, and CERNLIB's order is
# brittle; dropping libraries can come once M0 is linking cleanly.
target_link_dellibs(delphi-raw-nanoaod
    skelana phdst dstana pxdst vfclap vdclap ufield bsaurus herlib trigger uhlib dstana)
target_link_cernlibs(delphi-raw-nanoaod
    genlib packlib kernlib ariadne herwig jetset74)
target_link_libraries(delphi-raw-nanoaod gfortran)

target_link_dellibs(delphi-raw-bank-lister
    skelana phdst dstana pxdst vfclap vdclap ufield bsaurus herlib trigger uhlib dstana)
target_link_cernlibs(delphi-raw-bank-lister
    genlib packlib kernlib ariadne herwig jetset74)
target_link_libraries(delphi-raw-bank-lister gfortran)
