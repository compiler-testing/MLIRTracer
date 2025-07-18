module {
  func.func @main(%arg0: tensor<69x12x24x53x38xi8>, %arg1: tensor<51x75x70xf32>, %arg2: tensor<6xi1>) -> (tensor<3x5472x1219x2xi8>, tensor<51x75x70xf32>, tensor<1xi1>, tensor<1xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 3, 5472, 1219, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<69x12x24x53x38xi8>, !tosa.shape<4>) -> tensor<3x5472x1219x2xi8>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<3x5472x1219x2xi8>, tensor<3x5472x1219x2xi8>) -> tensor<3x5472x1219x2xi8>
    %2 = tosa.reciprocal %arg1 : (tensor<51x75x70xf32>) -> tensor<51x75x70xf32>
    %3 = tosa.exp %2 : (tensor<51x75x70xf32>) -> tensor<51x75x70xf32>
    %4 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.clz %4 : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %1, %3, %6, %7 : tensor<3x5472x1219x2xi8>, tensor<51x75x70xf32>, tensor<1xi1>, tensor<1xi1>
  }
}
