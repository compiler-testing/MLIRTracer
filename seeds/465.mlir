module {
  func.func @main(%arg0: tensor<62x59x67x65x25xi1>, %arg1: tensor<62x59x1x65x25xi1>, %arg2: tensor<20x28x6xi8>, %arg3: tensor<30x44x1x42xi1>, %arg4: tensor<87x42x97x77x13xf32>) -> (tensor<62x59x67x65x25xi1>, tensor<40x28x18xi8>, tensor<87x42x97x77x13xf32>, tensor<1x44x1x1xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<62x59x67x65x25xi1>, tensor<62x59x1x65x25xi1>) -> tensor<62x59x67x65x25xi1>
    %t_1 = tosa.const_shape {values = dense<[ 2, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg2, %t_1 : (tensor<20x28x6xi8>, !tosa.shape<3>) -> tensor<40x28x18xi8>
    %2 = tosa.reduce_any %arg3 {axis = 3 : i32} : (tensor<30x44x1x42xi1>) -> tensor<30x44x1x1xi1>
    %3 = tosa.sigmoid %arg4 : (tensor<87x42x97x77x13xf32>) -> tensor<87x42x97x77x13xf32>
    %4 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<30x44x1x1xi1>) -> tensor<1x44x1x1xi1>
    %5 = tosa.clz %4 : (tensor<1x44x1x1xi1>) -> tensor<1x44x1x1xi1>
    return %0, %1, %3, %5 : tensor<62x59x67x65x25xi1>, tensor<40x28x18xi8>, tensor<87x42x97x77x13xf32>, tensor<1x44x1x1xi1>
  }
}
