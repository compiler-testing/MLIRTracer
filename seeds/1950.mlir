module {
  func.func @main(%arg0: tensor<55x78x32x34xi32>, %arg1: tensor<55x1x32x34xi32>, %arg2: tensor<51x66x53xi1>, %arg3: tensor<51x1x53xi1>) -> (tensor<55x78x32x34xi32>, tensor<51x66x1xi1>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<55x78x32x34xi32>, tensor<55x1x32x34xi32>) -> tensor<55x78x32x34xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<55x78x32x34xi32>, tensor<55x78x32x34xi32>) -> tensor<55x78x32x34xi32>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<51x66x53xi1>, tensor<51x1x53xi1>) -> tensor<51x66x53xi1>
    %3 = tosa.reduce_max %2 {axis = 2 : i32} : (tensor<51x66x53xi1>) -> tensor<51x66x1xi1>
    return %1, %3 : tensor<55x78x32x34xi32>, tensor<51x66x1xi1>
  }
}
