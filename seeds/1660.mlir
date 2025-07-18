module {
  func.func @main(%arg0: tensor<76x75x76x26xi32>, %arg1: tensor<73x71x92x19xi1>) -> (tensor<76x75x76x52xi32>, tensor<73x71x92x1xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<76x75x76x26xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<76x75x76x26xi32>
    %1 = tosa.concat %0, %0 {axis = 3 : i32} : (tensor<76x75x76x26xi32>, tensor<76x75x76x26xi32>) -> tensor<76x75x76x52xi32>
    %2 = tosa.reduce_any %arg1 {axis = 3 : i32} : (tensor<73x71x92x19xi1>) -> tensor<73x71x92x1xi1>
    return %1, %2 : tensor<76x75x76x52xi32>, tensor<73x71x92x1xi1>
  }
}
