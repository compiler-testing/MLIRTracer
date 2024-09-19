## setup

### Build TosaGenerator
TosaGenerator is a generator that generate diverse tosa graph IR for MLIR testing, which is developed based on LLVM repository (git version eb601430d3d7f45c30ef8d793a45cbcedf910577).  
This setup assumes that you have built LLVM in `$BUILD_DIR`. To build generator, run
```
$ cd TosaGenerator
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
$ ninja
```

### Run fuzzing
To build and launch the tests, run
```
cd fuzz_tool
# Generate 500 Tosa IRs and initialize the seed pool
python3 ./src/main.py --opt=generator  --sqlName=test
# Run fuzzing
python3 ./src/main.py --opt=fuzz  --sqlName=test
```