module {
  func.func @main(%arg0: tensor<60xi8>, %arg1: tensor<1xi8>, %arg2: tensor<2x72x65xi1>, %arg3: tensor<2x1x1xi1>) -> (tensor<3x10x1x1xi1>, tensor<2x72x1xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<60xi8>, tensor<1xi8>) -> tensor<60xi8>
    %1 = tosa.sub %0, %0 : (tensor<60xi8>, tensor<60xi8>) -> tensor<60xi8>
    %r_2 = tosa.const_shape {values = dense<[ 3, 10, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<60xi8>, !tosa.shape<4>) -> tensor<3x10x1x2xi8>
    %3 = tosa.reduce_product %2 {axis = 3 : i32} : (tensor<3x10x1x2xi8>) -> tensor<3x10x1x1xi8>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<3x10x1x1xi8>, tensor<3x10x1x1xi8>) -> tensor<3x10x1x1xi8>
    %5 = tosa.logical_xor %arg2, %arg3 : (tensor<2x72x65xi1>, tensor<2x1x1xi1>) -> tensor<2x72x65xi1>
    %6 = tosa.arithmetic_right_shift %4, %4 {round = false} : (tensor<3x10x1x1xi8>, tensor<3x10x1x1xi8>) -> tensor<3x10x1x1xi8>
    %7 = tosa.greater_equal %6, %3 : (tensor<3x10x1x1xi8>, tensor<3x10x1x1xi8>) -> tensor<3x10x1x1xi1>
    %8 = tosa.reduce_max %5 {axis = 2 : i32} : (tensor<2x72x65xi1>) -> tensor<2x72x1xi1>
    return %7, %8 : tensor<3x10x1x1xi1>, tensor<2x72x1xi1>
  }
}
