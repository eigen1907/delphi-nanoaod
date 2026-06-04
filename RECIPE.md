# SIMPLE TEST

```bash
git clone https://github.com/eigen1907/delphi-nanoaod.git -b jshin-dev-fullDST
cd delphi-nanoaod
source setup.sh
cmake -B build
cmake --build build
bash sanity-check.sh
# run notebooks
# with 
conda env create -f environment.yml
```

```
python python/merge_raw_nanoaod.py \
  --sdst output/florian/20260602_SmallTest_Zmumu/final_root/job_0/nanoaod.root \
  --raw-sdst output/florian/20260602_SmallTest_Zmumu/final_root/job_0/nanoaod_raw_sdst.root \
  --raw-fadana output/florian/20260602_SmallTest_Zmumu/final_root/job_0/nanoaod_raw_fadana.root \
  --output output/florian/20260602_SmallTest_Zmumu/final_root/job_0/merged.root
```