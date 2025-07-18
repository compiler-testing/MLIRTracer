module {
  func.func @main(%arg0: tensor<56x35x74x33xf32>, %arg1: tensor<26xi1>, %arg2: tensor<26xi1>) -> (tensor<56x35x74x33xf32>, tensor<26xi1>, tensor<i32>) {
    %0 = tosa.tanh %arg0 : (tensor<56x35x74x33xf32>) -> tensor<56x35x74x33xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<26xi1>, tensor<26xi1>) -> tensor<26xi1>
    %2 = tosa.argmax %1 {axis = 0 : i32} : (tensor<26xi1>) -> tensor<i32>
    %3 = tosa.tanh %0 : (tensor<56x35x74x33xf32>) -> tensor<56x35x74x33xf32>
    %4 = tosa.logical_not %1 : (tensor<26xi1>) -> tensor<26xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = tosa.negate %2, %in_zp_5, %out_zp_5 : (tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    return %3, %4, %5 : tensor<56x35x74x33xf32>, tensor<26xi1>, tensor<i32>
  }
}
