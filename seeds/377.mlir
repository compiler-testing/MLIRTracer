module {
  func.func @main(%arg0: tensor<21x92x33x49xf32>, %arg1: tensor<21x92x33x17xf32>, %arg2: tensor<83xi64>) -> (tensor<83xi64>, tensor<21x92x33x66xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 3 : i32} : (tensor<21x92x33x49xf32>, tensor<21x92x33x17xf32>) -> tensor<21x92x33x66xf32>
    %1 = tosa.floor %0 : (tensor<21x92x33x66xf32>) -> tensor<21x92x33x66xf32>
    %2 = tosa.clz %arg2 : (tensor<83xi64>) -> tensor<83xi64>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<83xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<83xi64>
    %4 = tosa.reverse %1 {axis = 1 : i32} : (tensor<21x92x33x66xf32>) -> tensor<21x92x33x66xf32>
    return %3, %4 : tensor<83xi64>, tensor<21x92x33x66xf32>
  }
}
