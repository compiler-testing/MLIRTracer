module {
  func.func @main(%arg0: tensor<3x31x44xi32>, %arg1: tensor<3x31x44xi32>, %arg2: tensor<56x94x70x21xf32>) -> (tensor<3x31x44xi32>, tensor<56x94x70x21xi1>, tensor<56x94x70x21xf32>, tensor<31x1xi32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<3x31x44xi32>, tensor<3x31x44xi32>) -> tensor<3x31x44xi32>
    %1 = tosa.add %0, %0 : (tensor<3x31x44xi32>, tensor<3x31x44xi32>) -> tensor<3x31x44xi32>
    %2 = tosa.rsqrt %arg2 : (tensor<56x94x70x21xf32>) -> tensor<56x94x70x21xf32>
    %3 = tosa.greater %1, %0 : (tensor<3x31x44xi32>, tensor<3x31x44xi32>) -> tensor<3x31x44xi1>
    %4 = tosa.intdiv %1, %1 : (tensor<3x31x44xi32>, tensor<3x31x44xi32>) -> tensor<3x31x44xi32>
    %5 = tosa.logical_left_shift %3, %3 : (tensor<3x31x44xi1>, tensor<3x31x44xi1>) -> tensor<3x31x44xi1>
    %6 = tosa.reduce_any %5 {axis = 2 : i32} : (tensor<3x31x44xi1>) -> tensor<3x31x1xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<3x31x1xi1>, tensor<3x31x1xi1>) -> tensor<3x31x1xi1>
    %t_8 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.tile %7, %t_8 : (tensor<3x31x1xi1>, !tosa.shape<3>) -> tensor<3x31x1xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<3x31x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<3x31x1xi1>
    %10 = tosa.sub %9, %7 : (tensor<3x31x1xi1>, tensor<3x31x1xi1>) -> tensor<3x31x1xi1>
    %11 = tosa.argmax %10 {axis = 0 : i32} : (tensor<3x31x1xi1>) -> tensor<31x1xi32>
    %12 = tosa.intdiv %11, %11 : (tensor<31x1xi32>, tensor<31x1xi32>) -> tensor<31x1xi32>
    %13 = tosa.equal %2, %2 : (tensor<56x94x70x21xf32>, tensor<56x94x70x21xf32>) -> tensor<56x94x70x21xi1>
    %14 = tosa.tanh %2 : (tensor<56x94x70x21xf32>) -> tensor<56x94x70x21xf32>
    %15 = tosa.bitwise_xor %12, %11 : (tensor<31x1xi32>, tensor<31x1xi32>) -> tensor<31x1xi32>
    return %4, %13, %14, %15 : tensor<3x31x44xi32>, tensor<56x94x70x21xi1>, tensor<56x94x70x21xf32>, tensor<31x1xi32>
  }
}
