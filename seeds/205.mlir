module {
  func.func @main(%arg0: tensor<82x26x2x94x67xi8>, %arg1: tensor<1x1x2x94x67xi8>, %arg2: tensor<44x43x88x83x48x91xi1>, %arg3: tensor<44x1x1x1x1x1xi1>) -> (tensor<82x26x2x94x134xi8>, tensor<44x43x88x83x48x91xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<82x26x2x94x67xi8>, tensor<1x1x2x94x67xi8>) -> tensor<82x26x2x94x67xi8>
    %1 = tosa.bitwise_not %0 : (tensor<82x26x2x94x67xi8>) -> tensor<82x26x2x94x67xi8>
    %2 = tosa.sub %1, %1 : (tensor<82x26x2x94x67xi8>, tensor<82x26x2x94x67xi8>) -> tensor<82x26x2x94x67xi8>
    %3 = tosa.abs %2 : (tensor<82x26x2x94x67xi8>) -> tensor<82x26x2x94x67xi8>
    %4 = tosa.abs %3 : (tensor<82x26x2x94x67xi8>) -> tensor<82x26x2x94x67xi8>
    %5 = tosa.concat %4, %3 {axis = 4 : i32} : (tensor<82x26x2x94x67xi8>, tensor<82x26x2x94x67xi8>) -> tensor<82x26x2x94x134xi8>
    %6 = tosa.clz %5 : (tensor<82x26x2x94x134xi8>) -> tensor<82x26x2x94x134xi8>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = tosa.negate %6, %in_zp_7, %out_zp_7 : (tensor<82x26x2x94x134xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<82x26x2x94x134xi8>
    %8 = tosa.logical_xor %arg2, %arg3 : (tensor<44x43x88x83x48x91xi1>, tensor<44x1x1x1x1x1xi1>) -> tensor<44x43x88x83x48x91xi1>
    return %7, %8 : tensor<82x26x2x94x134xi8>, tensor<44x43x88x83x48x91xi1>
  }
}
