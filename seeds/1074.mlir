module {
  func.func @main(%arg0: tensor<14x55x98x100x62x2xi32>, %arg1: tensor<97x78x21x28xf32>, %arg2: tensor<37x52xi1>, %arg3: tensor<1x1xi1>) -> (tensor<37x52xi1>, tensor<5x2x7x7x4x1xi32>, tensor<78x21x28xi32>, tensor<156x21x84xi32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 7, 2, 5, 8, 10, 1 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 5, 2, 7, 7, 4, 1 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<14x55x98x100x62x2xi32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<5x2x7x7x4x1xi32>
    %1 = tosa.add %0, %0 : (tensor<5x2x7x7x4x1xi32>, tensor<5x2x7x7x4x1xi32>) -> tensor<5x2x7x7x4x1xi32>
    %2 = tosa.exp %arg1 : (tensor<97x78x21x28xf32>) -> tensor<97x78x21x28xf32>
    %3 = tosa.logical_and %arg2, %arg3 : (tensor<37x52xi1>, tensor<1x1xi1>) -> tensor<37x52xi1>
    %4 = tosa.argmax %2 {axis = 0 : i32} : (tensor<97x78x21x28xf32>) -> tensor<78x21x28xi32>
    %5 = tosa.bitwise_and %4, %4 : (tensor<78x21x28xi32>, tensor<78x21x28xi32>) -> tensor<78x21x28xi32>
    %6 = tosa.add %1, %1 : (tensor<5x2x7x7x4x1xi32>, tensor<5x2x7x7x4x1xi32>) -> tensor<5x2x7x7x4x1xi32>
    %7 = tosa.arithmetic_right_shift %5, %4 {round = true} : (tensor<78x21x28xi32>, tensor<78x21x28xi32>) -> tensor<78x21x28xi32>
    %t_8 = tosa.const_shape {values = dense<[ 2, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.tile %5, %t_8 : (tensor<78x21x28xi32>, !tosa.shape<3>) -> tensor<156x21x84xi32>
    return %3, %6, %7, %8 : tensor<37x52xi1>, tensor<5x2x7x7x4x1xi32>, tensor<78x21x28xi32>, tensor<156x21x84xi32>
  }
}
