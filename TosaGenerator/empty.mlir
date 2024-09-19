func.func @main() {
  return
}

//build/bin/mlirfuzzer-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" test.mlir
//build/bin/mlirfuzzer-opt empty.mlir -tosaGen
//build/bin/mlirfuzzer-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))"  test.mlir