module {
  func.func @main(%arg0: tensor<63x4x56xi1>, %arg1: tensor<1x1x1xi1>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<89x61x25xi64>, %arg5: tensor<1x61x25xi64>, %arg6: tensor<83xf32>) -> (tensor<i1>, tensor<63x4x56xi1>, tensor<89x61x25xi64>, tensor<83xf32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<63x4x56xi1>, tensor<1x1x1xi1>) -> tensor<63x4x56xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.add %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.bitwise_and %0, %0 : (tensor<63x4x56xi1>, tensor<63x4x56xi1>) -> tensor<63x4x56xi1>
    %5 = tosa.maximum %arg4, %arg5 : (tensor<89x61x25xi64>, tensor<1x61x25xi64>) -> tensor<89x61x25xi64>
    %6 = tosa.ceil %arg6 : (tensor<83xf32>) -> tensor<83xf32>
    %7 = tosa.exp %6 : (tensor<83xf32>) -> tensor<83xf32>
    return %3, %4, %5, %7 : tensor<i1>, tensor<63x4x56xi1>, tensor<89x61x25xi64>, tensor<83xf32>
  }
}
