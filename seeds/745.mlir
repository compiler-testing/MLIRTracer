module {
  func.func @main(%arg0: tensor<82x55x92x32x63x36xi16>, %arg1: tensor<82x55x1x32x1x36xi16>, %arg2: tensor<73x76xf32>) -> (tensor<82x55x92x32x63x36xi16>, tensor<73xi32>, tensor<219x76xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<82x55x92x32x63x36xi16>, tensor<82x55x1x32x1x36xi16>) -> tensor<82x55x92x32x63x36xi16>
    %1 = tosa.floor %arg2 : (tensor<73x76xf32>) -> tensor<73x76xf32>
    %2 = tosa.argmax %1 {axis = 1 : i32} : (tensor<73x76xf32>) -> tensor<73xi32>
    %3 = tosa.intdiv %2, %2 : (tensor<73xi32>, tensor<73xi32>) -> tensor<73xi32>
    %4 = tosa.add %3, %3 : (tensor<73xi32>, tensor<73xi32>) -> tensor<73xi32>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<73xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<73xi32>
    %t_6 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %1, %t_6 : (tensor<73x76xf32>, !tosa.shape<2>) -> tensor<219x76xf32>
    return %0, %5, %6 : tensor<82x55x92x32x63x36xi16>, tensor<73xi32>, tensor<219x76xf32>
  }
}
