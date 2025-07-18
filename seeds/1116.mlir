module {
  func.func @main(%arg0: tensor<40x17x75x34xi32>, %arg1: tensor<1x17x1x34xi32>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<62x20x1x84x5x92xf32>) -> (tensor<40x34x75x34xi32>, tensor<i1>, tensor<62x20x1x84x5x92xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<40x17x75x34xi32>, tensor<1x17x1x34xi32>) -> tensor<40x17x75x34xi32>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<40x17x75x34xi32>, tensor<40x17x75x34xi32>) -> tensor<40x17x75x34xi32>
    %2 = tosa.logical_or %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %t_3 = tosa.const_shape {values = dense<[ 1, 2, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.tile %1, %t_3 : (tensor<40x17x75x34xi32>, !tosa.shape<4>) -> tensor<40x34x75x34xi32>
    %4 = tosa.reverse %3 {axis = 1 : i32} : (tensor<40x34x75x34xi32>) -> tensor<40x34x75x34xi32>
    %5 = tosa.sub %4, %4 : (tensor<40x34x75x34xi32>, tensor<40x34x75x34xi32>) -> tensor<40x34x75x34xi32>
    %6 = tosa.bitwise_xor %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.clz %5 : (tensor<40x34x75x34xi32>) -> tensor<40x34x75x34xi32>
    %8 = tosa.abs %6 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.sigmoid %arg4 : (tensor<62x20x1x84x5x92xf32>) -> tensor<62x20x1x84x5x92xf32>
    return %7, %8, %9 : tensor<40x34x75x34xi32>, tensor<i1>, tensor<62x20x1x84x5x92xf32>
  }
}
