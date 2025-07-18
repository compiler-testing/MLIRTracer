module {
  func.func @main(%arg0: tensor<98x23x68x93x79x41xi64>, %arg1: tensor<98x23x68x1x79x41xi64>, %arg2: tensor<31x34x52xi8>) -> (tensor<98x23x68x93x79x41xi64>, tensor<1x1x52xi8>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<98x23x68x93x79x41xi64>, tensor<98x23x68x1x79x41xi64>) -> tensor<98x23x68x93x79x41xi64>
    %1 = tosa.reduce_product %arg2 {axis = 1 : i32} : (tensor<31x34x52xi8>) -> tensor<31x1x52xi8>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<31x1x52xi8>) -> tensor<1x1x52xi8>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<1x1x52xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x52xi8>
    return %0, %3 : tensor<98x23x68x93x79x41xi64>, tensor<1x1x52xi8>
  }
}
