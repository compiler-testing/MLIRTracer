module {
  func.func @main(%arg0: tensor<56x53xi16>, %arg1: tensor<94x33xi1>, %arg2: tensor<78x4x4x77x53xf32>) -> (tensor<112xi32>, tensor<94x1xi1>, tensor<78x4x4x77x53xf32>, tensor<94x1xi1>, tensor<1x2x47x1xi1>, tensor<94x1x1xi1>) {
    %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<56x53xi16>) -> tensor<56xi32>
    %1 = tosa.clamp %0 {min_val = -27 : i32, max_val = -10 : i32} : (tensor<56xi32>) -> tensor<56xi32>
    %2 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<94x33xi1>) -> tensor<94x1xi1>
    %3 = tosa.tanh %arg2 : (tensor<78x4x4x77x53xf32>) -> tensor<78x4x4x77x53xf32>
    %t_4 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %1, %t_4 : (tensor<56xi32>, !tosa.shape<1>) -> tensor<112xi32>
    %5 = tosa.sigmoid %3 : (tensor<78x4x4x77x53xf32>) -> tensor<78x4x4x77x53xf32>
    %6 = tosa.pow %5, %3 : (tensor<78x4x4x77x53xf32>, tensor<78x4x4x77x53xf32>) -> tensor<78x4x4x77x53xf32>
    %7 = tosa.logical_not %2 : (tensor<94x1xi1>) -> tensor<94x1xi1>
    %8 = tosa.bitwise_not %2 : (tensor<94x1xi1>) -> tensor<94x1xi1>
    %9 = tosa.reduce_product %8 {axis = 1 : i32} : (tensor<94x1xi1>) -> tensor<94x1xi1>
    %10 = tosa.reduce_sum %7 {axis = 1 : i32} : (tensor<94x1xi1>) -> tensor<94x1xi1>
    %11 = tosa.add %10, %7 : (tensor<94x1xi1>, tensor<94x1xi1>) -> tensor<94x1xi1>
    %12 = tosa.minimum %3, %6 : (tensor<78x4x4x77x53xf32>, tensor<78x4x4x77x53xf32>) -> tensor<78x4x4x77x53xf32>
    %13 = tosa.bitwise_xor %9, %10 : (tensor<94x1xi1>, tensor<94x1xi1>) -> tensor<94x1xi1>
    %14 = tosa.bitwise_not %13 : (tensor<94x1xi1>) -> tensor<94x1xi1>
    %r_15 = tosa.const_shape {values = dense<[ 1, 2, 47, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %15 = tosa.reshape %9, %r_15 : (tensor<94x1xi1>, !tosa.shape<4>) -> tensor<1x2x47x1xi1>
    %r_16 = tosa.const_shape {values = dense<[ 94, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %16 = tosa.reshape %9, %r_16 : (tensor<94x1xi1>, !tosa.shape<3>) -> tensor<94x1x1xi1>
    return %4, %11, %12, %14, %15, %16 : tensor<112xi32>, tensor<94x1xi1>, tensor<78x4x4x77x53xf32>, tensor<94x1xi1>, tensor<1x2x47x1xi1>, tensor<94x1x1xi1>
  }
}
