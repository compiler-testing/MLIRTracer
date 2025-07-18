module {
  func.func @main(%arg0: tensor<21x16x20xi32>, %arg1: tensor<17x83xf32>, %arg2: tensor<10xi1>, %arg3: tensor<1xi1>) -> (tensor<3x1x7xi32>, tensor<17x83xf32>, tensor<10xi1>) {
    %s_0_start = tosa.const_shape {values = dense<[ 18, 15, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 3, 1, 7 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<21x16x20xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x1x7xi32>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<3x1x7xi32>) -> tensor<3x1x7xi32>
    %2 = tosa.clamp %1 {min_val = -64 : i32, max_val = -54 : i32} : (tensor<3x1x7xi32>) -> tensor<3x1x7xi32>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<3x1x7xi32>) -> tensor<3x1x7xi32>
    %4 = tosa.sigmoid %arg1 : (tensor<17x83xf32>) -> tensor<17x83xf32>
    %5 = tosa.exp %4 : (tensor<17x83xf32>) -> tensor<17x83xf32>
    %6 = tosa.logical_or %arg2, %arg3 : (tensor<10xi1>, tensor<1xi1>) -> tensor<10xi1>
    return %3, %5, %6 : tensor<3x1x7xi32>, tensor<17x83xf32>, tensor<10xi1>
  }
}
