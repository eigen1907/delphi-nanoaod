source setup.sh

cmake -B build
cmake --build build -j 4

mkdir -p output/sanity-check/{nanoaod,raw-nanoaod,plots}

./build/delphi-nanoaod/delphi-nanoaod \
  --nickname short94_c2 \
  --config config/delphi-nanoaod.yaml \
  --output output/sanity-check/nanoaod/short94_c2.root \
  -m 100 > output/sanity-check/nanoaod/short94_c2.log 2>&1

./build/delphi-raw-nanoaod/delphi-raw-nanoaod \
  --nickname short94_c2 \
  --output output/sanity-check/raw-nanoaod/short94_c2.root \
  -m 100 > output/sanity-check/raw-nanoaod/short94_c2.log 2>&1

./build/delphi-nanoaod/delphi-nanoaod \
  --nickname sh_qqps_e91.25_c94_2l_c2 \
  --mc \
  --config config/delphi-nanoaod.yaml \
  --output output/sanity-check/nanoaod/sh_qqps_e91.25_c94_2l_c2.root \
  -m 100 > output/sanity-check/nanoaod/sh_qqps_e91.25_c94_2l_c2.log 2>&1

./build/delphi-raw-nanoaod/delphi-raw-nanoaod \
  --nickname sh_qqps_e91.25_c94_2l_c2 \
  --output output/sanity-check/raw-nanoaod/sh_qqps_e91.25_c94_2l_c2.root \
  -m 100 > output/sanity-check/raw-nanoaod/sh_qqps_e91.25_c94_2l_c2.log 2>&1
