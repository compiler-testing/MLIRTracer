module {
  func.func @main(%arg0: tensor<60x10x56x60xi1>, %arg1: tensor<60x10x1x1xi1>, %arg2: tensor<20x24x39xi32>, %arg3: tensor<1x24x39xi32>, %arg4: tensor<f32>) -> (tensor<3120x3x2x1xi32>, tensor<f32>, tensor<20x24x39xi32>, tensor<28x1x2xi1>, tensor<20x1x39xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<60x10x56x60xi1>, tensor<60x10x1x1xi1>) -> tensor<60x10x56x60xi1>
    %1 = tosa.reduce_min %0 {axis = 3 : i32} : (tensor<60x10x56x60xi1>) -> tensor<60x10x56x1xi1>
    %2 = tosa.maximum %arg2, %arg3 : (tensor<20x24x39xi32>, tensor<1x24x39xi32>) -> tensor<20x24x39xi32>
    %3 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<60x10x56x1xi1>) -> tensor<60x1x56x1xi1>
    %4 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<60x1x56x1xi1>) -> tensor<1x1x56x1xi1>
    %5 = tosa.sub %2, %2 : (tensor<20x24x39xi32>, tensor<20x24x39xi32>) -> tensor<20x24x39xi32>
    %6 = tosa.rsqrt %arg4 : (tensor<f32>) -> tensor<f32>
    %r_7 = tosa.const_shape {values = dense<[ 28, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.reshape %4, %r_7 : (tensor<1x1x56x1xi1>, !tosa.shape<3>) -> tensor<28x1x2xi1>
    %8 = tosa.reduce_any %7 {axis = 1 : i32} : (tensor<28x1x2xi1>) -> tensor<28x1x2xi1>
    %r_9 = tosa.const_shape {values = dense<[ 3120, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.reshape %5, %r_9 : (tensor<20x24x39xi32>, !tosa.shape<4>) -> tensor<3120x3x2x1xi32>
    %10 = tosa.rsqrt %6 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.clamp %5 {min_val = 25 : i32, max_val = 131 : i32} : (tensor<20x24x39xi32>) -> tensor<20x24x39xi32>
    %12 = tosa.logical_xor %8, %7 : (tensor<28x1x2xi1>, tensor<28x1x2xi1>) -> tensor<28x1x2xi1>
    %13 = tosa.sub %12, %8 : (tensor<28x1x2xi1>, tensor<28x1x2xi1>) -> tensor<28x1x2xi1>
    %14 = tosa.bitwise_xor %11, %2 : (tensor<20x24x39xi32>, tensor<20x24x39xi32>) -> tensor<20x24x39xi32>
    %15 = tosa.logical_xor %13, %7 : (tensor<28x1x2xi1>, tensor<28x1x2xi1>) -> tensor<28x1x2xi1>
    %16 = tosa.reduce_sum %11 {axis = 1 : i32} : (tensor<20x24x39xi32>) -> tensor<20x1x39xi32>
    %17 = tosa.greater %16, %16 : (tensor<20x1x39xi32>, tensor<20x1x39xi32>) -> tensor<20x1x39xi1>
    return %9, %10, %14, %15, %17 : tensor<3120x3x2x1xi32>, tensor<f32>, tensor<20x24x39xi32>, tensor<28x1x2xi1>, tensor<20x1x39xi1>
  }
}
