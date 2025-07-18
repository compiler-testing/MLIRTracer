module {
  func.func @main(%arg0: tensor<67x57x96xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<55x54xi1>, %arg3: tensor<1x1xi1>) -> (tensor<67x57x96xf32>, tensor<55x54xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<67x57x96xf32>, tensor<1x1x1xf32>) -> tensor<67x57x96xf32>
    %1 = tosa.clamp %0 {min_val = 4.200000e+01 : f32, max_val = 5.300000e+01 : f32} : (tensor<67x57x96xf32>) -> tensor<67x57x96xf32>
    %2 = tosa.bitwise_or %arg2, %arg3 : (tensor<55x54xi1>, tensor<1x1xi1>) -> tensor<55x54xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<55x54xi1>, tensor<55x54xi1>) -> tensor<55x54xi1>
    return %1, %3 : tensor<67x57x96xf32>, tensor<55x54xi1>
  }
}
