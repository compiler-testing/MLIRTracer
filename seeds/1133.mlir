module {
  func.func @main(%arg0: tensor<8xi8>, %arg1: tensor<1xi8>, %arg2: tensor<49x20xi1>, %arg3: tensor<f32>) -> (tensor<3x60xi1>, tensor<f32>, tensor<1xi8>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<8xi8>, tensor<1xi8>) -> tensor<8xi8>
    %1 = tosa.maximum %0, %0 : (tensor<8xi8>, tensor<8xi8>) -> tensor<8xi8>
    %2 = tosa.bitwise_xor %1, %0 : (tensor<8xi8>, tensor<8xi8>) -> tensor<8xi8>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<8xi8>) -> tensor<1xi8>
    %4 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<49x20xi1>) -> tensor<1x20xi1>
    %t_5 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %4, %t_5 : (tensor<1x20xi1>, !tosa.shape<2>) -> tensor<3x60xi1>
    %6 = tosa.exp %arg3 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.bitwise_and %3, %3 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %t_8 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.tile %7, %t_8 : (tensor<1xi8>, !tosa.shape<1>) -> tensor<1xi8>
    return %5, %6, %8 : tensor<3x60xi1>, tensor<f32>, tensor<1xi8>
  }
}
