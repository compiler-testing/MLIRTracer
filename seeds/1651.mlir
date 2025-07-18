module {
  func.func @main(%arg0: tensor<98x70x98x31x67xf32>, %arg1: tensor<93x16x27xi1>, %arg2: tensor<93x16x27xi1>) -> (tensor<98x70x98x31x67xi1>, tensor<93x27x16xi1>, tensor<98x70x98x31x67xf32>, tensor<98x70x98x31x67xf32>, tensor<98x70x98x31x67xf32>, tensor<93x1x27xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xf32>
    %1 = tosa.abs %0 : (tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xf32>
    %2 = tosa.bitwise_and %arg1, %arg2 : (tensor<93x16x27xi1>, tensor<93x16x27xi1>) -> tensor<93x16x27xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<93x16x27xi1>, tensor<93x16x27xi1>) -> tensor<93x16x27xi1>
    %4 = tosa.reverse %2 {axis = 2 : i32} : (tensor<93x16x27xi1>) -> tensor<93x16x27xi1>
    %5 = tosa.equal %0, %1 : (tensor<98x70x98x31x67xf32>, tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xi1>
    %6 = tosa.logical_not %3 : (tensor<93x16x27xi1>) -> tensor<93x16x27xi1>
    %7 = tosa.reduce_any %6 {axis = 1 : i32} : (tensor<93x16x27xi1>) -> tensor<93x1x27xi1>
    %8 = tosa.floor %0 : (tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xf32>
    %9 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %10 = tosa.transpose %4 {perms = array<i32: 0, 2, 1>} : (tensor<93x16x27xi1>) -> tensor<93x27x16xi1>
    %11 = tosa.clz %7 : (tensor<93x1x27xi1>) -> tensor<93x1x27xi1>
    %12 = tosa.minimum %8, %1 : (tensor<98x70x98x31x67xf32>, tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xf32>
    %13 = tosa.rsqrt %1 : (tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xf32>
    %14 = tosa.reciprocal %0 : (tensor<98x70x98x31x67xf32>) -> tensor<98x70x98x31x67xf32>
    %15 = tosa.logical_right_shift %11, %11 : (tensor<93x1x27xi1>, tensor<93x1x27xi1>) -> tensor<93x1x27xi1>
    return %5, %10, %12, %13, %14, %15 : tensor<98x70x98x31x67xi1>, tensor<93x27x16xi1>, tensor<98x70x98x31x67xf32>, tensor<98x70x98x31x67xf32>, tensor<98x70x98x31x67xf32>, tensor<93x1x27xi1>
  }
}
