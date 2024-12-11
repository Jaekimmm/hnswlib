### Test with Random-generated data
#### update_gen_data.py
Generate random-number dataset as user-defined config. (incl. data/query/gt/)
data config cannot be set in the file. (N, dummy_data_multiplier, N_queries, d, K)
Generated random number were normalized

#### updates_test.cpp
if update, rebuild graph db with the data in ../tests/cpp/data/
and do test


### Test with SIFT dataset ()
#### download_bigann.py
Download bigann vector files from original ftp to {hnswlib}/bigann
#### sift_test.cpp
ignore
#### sift_1b.cpp
- Set test config (subset_size_millions, efConstruction, M)
- Prepared dataset : 1M, 2M, 5M, 10M, 20M, 50M, 100M, 200M, 500M 1B) 

### Test with other SIFT size
- need to generate dataset using modified update_gen_data.py & update_test.cpp
