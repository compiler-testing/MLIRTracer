module {
  func.func @main(%arg0: tensor<7x12x49x73xi32>, %arg1: tensor<f32>, %arg2: tensor<3xi1>, %arg3: tensor<3xi1>) -> (tensor<3xi1>, tensor<i1>, tensor<5x4x2x1xi32>, tensor<1xi1>, tensor<7x1x49x73xi32>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<7x12x49x73xi32>) -> tensor<7x1x49x73xi32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<7x1x49x73xi32>) -> tensor<7x1x49x73xi32>
    %r_2 = tosa.const_shape {values = dense<[ 3577, 1, 1, 7 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<7x1x49x73xi32>, !tosa.shape<4>) -> tensor<3577x1x1x7xi32>
    %3 = "tosa.const"() {values = dense<[2, 3, 1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 1, 3, 2, 0>} : (tensor<3577x1x1x7xi32>) -> tensor<1x7x1x3577xi32>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 1, 0, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 5, 4, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<1x7x1x3577xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x4x2x1xi32>
    %6 = tosa.log %arg1 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.clz %5 : (tensor<5x4x2x1xi32>) -> tensor<5x4x2x1xi32>
    %8 = tosa.logical_xor %arg2, %arg3 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    %9 = tosa.logical_or %8, %8 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    %10 = tosa.greater %6, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = tosa.logical_left_shift %8, %8 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    %12 = tosa.maximum %7, %5 : (tensor<5x4x2x1xi32>, tensor<5x4x2x1xi32>) -> tensor<5x4x2x1xi32>
    %13 = tosa.reduce_sum %11 {axis = 0 : i32} : (tensor<3xi1>) -> tensor<1xi1>
    %14 = tosa.maximum %1, %0 : (tensor<7x1x49x73xi32>, tensor<7x1x49x73xi32>) -> tensor<7x1x49x73xi32>
    return %9, %10, %12, %13, %14 : tensor<3xi1>, tensor<i1>, tensor<5x4x2x1xi32>, tensor<1xi1>, tensor<7x1x49x73xi32>
  }
}
