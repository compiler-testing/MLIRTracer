module {
  func.func @main(%arg0: tensor<63xi8>, %arg1: tensor<63xi8>, %arg2: tensor<90x87xi1>, %arg3: tensor<38x80x31x4x51x100xf32>, %arg4: tensor<38x80x1x1x1x1xf32>) -> (tensor<1xi8>, tensor<1x87xi1>, tensor<38x80x31x4x51x100xf32>, tensor<1x87xi1>, tensor<1x87xi1>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<63xi8>, tensor<63xi8>) -> tensor<63xi8>
    %1 = tosa.clz %0 : (tensor<63xi8>) -> tensor<63xi8>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<63xi8>) -> tensor<1xi8>
    %3 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<90x87xi1>) -> tensor<1x87xi1>
    %4 = tosa.pow %arg3, %arg4 : (tensor<38x80x31x4x51x100xf32>, tensor<38x80x1x1x1x1xf32>) -> tensor<38x80x31x4x51x100xf32>
    %5 = tosa.logical_or %3, %3 : (tensor<1x87xi1>, tensor<1x87xi1>) -> tensor<1x87xi1>
    %6 = tosa.tanh %4 : (tensor<38x80x31x4x51x100xf32>) -> tensor<38x80x31x4x51x100xf32>
    %7 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<1x87xi1>) -> tensor<1x87xi1>
    %8 = tosa.logical_xor %3, %3 : (tensor<1x87xi1>, tensor<1x87xi1>) -> tensor<1x87xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %7, %in_zp_9, %out_zp_9 : (tensor<1x87xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x87xi1>
    return %2, %5, %6, %8, %9 : tensor<1xi8>, tensor<1x87xi1>, tensor<38x80x31x4x51x100xf32>, tensor<1x87xi1>, tensor<1x87xi1>
  }
}
