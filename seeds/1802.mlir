module {
  func.func @main(%arg0: tensor<41x36x52x65x30xi1>, %arg1: tensor<1x1x1x1x30xi1>, %arg2: tensor<58x42xi8>, %arg3: tensor<28x10x68x19x39xf32>, %arg4: tensor<33x49x7x85xi1>) -> (tensor<1x42xi8>, tensor<28x10x68x19x39xf32>, tensor<41x36x52x65x30xi1>, tensor<105x187x1xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<41x36x52x65x30xi1>, tensor<1x1x1x1x30xi1>) -> tensor<41x36x52x65x30xi1>
    %1 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<58x42xi8>) -> tensor<1x42xi8>
    %2 = tosa.sigmoid %arg3 : (tensor<28x10x68x19x39xf32>) -> tensor<28x10x68x19x39xf32>
    %3 = tosa.clz %0 : (tensor<41x36x52x65x30xi1>) -> tensor<41x36x52x65x30xi1>
    %4 = tosa.reduce_all %arg4 {axis = 1 : i32} : (tensor<33x49x7x85xi1>) -> tensor<33x1x7x85xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<41x36x52x65x30xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<41x36x52x65x30xi1>
    %r_6 = tosa.const_shape {values = dense<[ 105, 187, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %4, %r_6 : (tensor<33x1x7x85xi1>, !tosa.shape<3>) -> tensor<105x187x1xi1>
    return %1, %2, %5, %6 : tensor<1x42xi8>, tensor<28x10x68x19x39xf32>, tensor<41x36x52x65x30xi1>, tensor<105x187x1xi1>
  }
}
