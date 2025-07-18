## Introduction

This is an implementation of the MLIR fuzzing framework described in: "Directed Testing in MLIR: Unleashing Its Potential by Overcoming the Limitations of Random Fuzzing"

MLIRTracer is a top-down fuzzing approach for MLIR, which systematically explores MLIR’s hierarchical code space following its design philosophy, while achieving directedness to enhance fuzzing efficiency.

April 2025 - The paper was accepted to FSE!



## Code Structure 
├── TosaGenerator        <- Code for generating TOSA IR.  
├── fuzz_tool            <- Implementation of the fuzzing process for testing.    
├── seeds                <- Available seeds.  
├── Motivation_example   <- The full example of motivation discussed in the paper.  
├── README.md            <- The top-level README for developers using this project.

## Setup
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 1) Set you config
The configuration file is located at: `./MLIRTracer/fuzz_tool/conf/conf.yml`.  
To run this project, you must modify this file to set your database information and the path of `mlir-opt`.
Additionally, this file provides default experiment settings, including the experiment duration and the number of seeds. You can adjust the experiment parameters as needed.

### 2) Generate or load seeds
#### Generate seeds with TosaGenerator
TosaGenerator is a generator that generate diverse tosa graph IR for MLIR testing, which is developed based on LLVM repository (git version [eb60143](https://github.com/llvm/llvm-project/commit/eb601430d3d7f45c30ef8d793a45cbcedf910577)).  
This setup assumes that you have built LLVM in `$BUILD_DIR`. To build generator, run
```
$ cd TosaGenerator
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
$ ninja
```
To generate Tosa IRs and initialize the seed pool, follow these steps:
```
cd fuzz_tool
python3 ./src/main.py --opt=generator  --sqlName=mlirfuzzing
```

#### or Directly load the available seed
Due to the rapid updates in the MLIR version, the TosaGenerator in this repository may no longer be compatible. We recommend using the test cases we have provided, which can run properly on the latest LLVM (git version [60579ec](https://github.com/llvm/llvm-project/commit/60579ec3059b2b6cc9dad90eaac1ed363fc395a7)). The command to load the seed is as follows:
```
cd fuzz_tool
python3 ./src/main.py --opt=load  --sqlName=mlirfuzzing
```
You can also use your own seed (tosa IR). Save your seed in `./MLIRTracer/seeds` and then execute the seed loading command above.


### Run fuzzing
The experimental version of LLVM progect in the paper is older (git version [eb60143](https://github.com/llvm/llvm-project/commit/eb601430d3d7f45c30ef8d793a45cbcedf910577)). 
We have updated the code, and it now supports MLIR testing on the latest LLVM (git version [60579ec](https://github.com/llvm/llvm-project/commit/60579ec3059b2b6cc9dad90eaac1ed363fc395a7)).

To build and launch the tests, run
```
cd fuzz_tool
python3 ./src/main.py --opt=fuzz  --sqlName=mlirfuzzing
```

## Contact
Weiyuan Tong - wytong@stumail.nwu.edu.cn

## Citation
```angular2html
@article{tong2025directed,
  title={Directed Testing in MLIR: Unleashing Its Potential by Overcoming the Limitations of Random Fuzzing},
  author={Tong, Weiyuan and Wang, Zixu and Tang, Zhanyong and Fang, Jianbin and Zhang, Yuqun and Ye, Guixin},
  journal={Proceedings of the ACM on Software Engineering},
  volume={2},
  number={FSE},
  pages={2288--2310},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```