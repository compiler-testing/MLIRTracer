module {
  func.func @main(%arg0: tensor<4x53x80xi64>, %arg1: tensor<4x80x94xi64>, %arg2: tensor<42x20x84x18x94x6xf32>, %arg3: tensor<16x29x76x94x59x90xi1>, %arg4: tensor<16x29x1x94x1x1xi1>) -> (tensor<4x53x1xi64>, tensor<42x20x84x18x94x6xf32>, tensor<4x106x188xi64>, tensor<42x20x84x18x94x6xf32>, tensor<42x20x84x18x94x6xf32>, tensor<16x29x76x94x59x90xi1>, tensor<5x9x7x1x1x5xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<4x53x80xi64>, tensor<4x80x94xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x53x94xi64>
    %1 = tosa.reciprocal %arg2 : (tensor<42x20x84x18x94x6xf32>) -> tensor<42x20x84x18x94x6xf32>
    %2 = tosa.bitwise_not %0 : (tensor<4x53x94xi64>) -> tensor<4x53x94xi64>
    %3 = tosa.tanh %1 : (tensor<42x20x84x18x94x6xf32>) -> tensor<42x20x84x18x94x6xf32>
    %4 = tosa.logical_or %arg3, %arg4 : (tensor<16x29x76x94x59x90xi1>, tensor<16x29x1x94x1x1xi1>) -> tensor<16x29x76x94x59x90xi1>
    %5 = tosa.reduce_max %0 {axis = 2 : i32} : (tensor<4x53x94xi64>) -> tensor<4x53x1xi64>
    %6 = tosa.logical_xor %4, %4 : (tensor<16x29x76x94x59x90xi1>, tensor<16x29x76x94x59x90xi1>) -> tensor<16x29x76x94x59x90xi1>
    %7 = tosa.add %4, %4 : (tensor<16x29x76x94x59x90xi1>, tensor<16x29x76x94x59x90xi1>) -> tensor<16x29x76x94x59x90xi1>
    %8 = tosa.log %1 : (tensor<42x20x84x18x94x6xf32>) -> tensor<42x20x84x18x94x6xf32>
    %t_9 = tosa.const_shape {values = dense<[ 1, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.tile %2, %t_9 : (tensor<4x53x94xi64>, !tosa.shape<3>) -> tensor<4x106x188xi64>
    %10 = tosa.rsqrt %1 : (tensor<42x20x84x18x94x6xf32>) -> tensor<42x20x84x18x94x6xf32>
    %11 = tosa.tanh %10 : (tensor<42x20x84x18x94x6xf32>) -> tensor<42x20x84x18x94x6xf32>
    %s_12_start = tosa.const_shape {values = dense<[ 7, 2, 14, 15, 11, 12 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_12_size = tosa.const_shape {values = dense<[ 5, 9, 7, 1, 1, 5 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %12 = tosa.slice %6, %s_12_start, %s_12_size : (tensor<16x29x76x94x59x90xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<5x9x7x1x1x5xi1>
    %13 = tosa.bitwise_or %7, %4 : (tensor<16x29x76x94x59x90xi1>, tensor<16x29x76x94x59x90xi1>) -> tensor<16x29x76x94x59x90xi1>
    %14 = tosa.reciprocal %3 : (tensor<42x20x84x18x94x6xf32>) -> tensor<42x20x84x18x94x6xf32>
    %15 = tosa.logical_and %13, %6 : (tensor<16x29x76x94x59x90xi1>, tensor<16x29x76x94x59x90xi1>) -> tensor<16x29x76x94x59x90xi1>
    %16 = tosa.logical_not %12 : (tensor<5x9x7x1x1x5xi1>) -> tensor<5x9x7x1x1x5xi1>
    return %5, %8, %9, %11, %14, %15, %16 : tensor<4x53x1xi64>, tensor<42x20x84x18x94x6xf32>, tensor<4x106x188xi64>, tensor<42x20x84x18x94x6xf32>, tensor<42x20x84x18x94x6xf32>, tensor<16x29x76x94x59x90xi1>, tensor<5x9x7x1x1x5xi1>
  }
}
