module {
  func.func @main(%arg0: tensor<79x75x5x56x15x22xf32>, %arg1: tensor<43x100x39x54xi16>, %arg2: tensor<43x100x39x1xi16>, %arg3: tensor<61x77x33x16xi1>) -> (tensor<43x100x39x1xi16>, tensor<39x54x100x43xi16>, tensor<1x77x33x16xi1>, tensor<4x4x7x12x12x12xi1>, tensor<4x4x7x12x12x12xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 53, 34, 0, 28, 1, 6 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_0_size = tosa.const_shape {values = dense<[ 4, 4, 7, 12, 12, 12 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<79x75x5x56x15x22xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x4x7x12x12x12xf32>
    %1 = tosa.abs %0 : (tensor<4x4x7x12x12x12xf32>) -> tensor<4x4x7x12x12x12xf32>
    %2 = tosa.arithmetic_right_shift %arg1, %arg2 {round = true} : (tensor<43x100x39x54xi16>, tensor<43x100x39x1xi16>) -> tensor<43x100x39x54xi16>
    %3 = tosa.reduce_max %2 {axis = 3 : i32} : (tensor<43x100x39x54xi16>) -> tensor<43x100x39x1xi16>
    %4 = "tosa.const"() {values = dense<[2, 3, 1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    %5 = tosa.transpose %2 {perms = array<i32: 2, 3, 1, 0>} : (tensor<43x100x39x54xi16>) -> tensor<39x54x100x43xi16>
    %6 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<61x77x33x16xi1>) -> tensor<1x77x33x16xi1>
    %7 = tosa.greater %1, %1 : (tensor<4x4x7x12x12x12xf32>, tensor<4x4x7x12x12x12xf32>) -> tensor<4x4x7x12x12x12xi1>
    %8 = tosa.greater %0, %0 : (tensor<4x4x7x12x12x12xf32>, tensor<4x4x7x12x12x12xf32>) -> tensor<4x4x7x12x12x12xi1>
    return %3, %5, %6, %7, %8 : tensor<43x100x39x1xi16>, tensor<39x54x100x43xi16>, tensor<1x77x33x16xi1>, tensor<4x4x7x12x12x12xi1>, tensor<4x4x7x12x12x12xi1>
  }
}
