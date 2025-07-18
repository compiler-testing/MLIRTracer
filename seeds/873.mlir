module {
  func.func @main(%arg0: tensor<34x83xf32>) -> tensor<2822xf32> {
    %0 = tosa.log %arg0 : (tensor<34x83xf32>) -> tensor<34x83xf32>
    %1 = tosa.clamp %0 {min_val = 1.000000e+01 : f32, max_val = 8.000000e+01 : f32} : (tensor<34x83xf32>) -> tensor<34x83xf32>
    %2 = tosa.rsqrt %1 : (tensor<34x83xf32>) -> tensor<34x83xf32>
    %3 = tosa.reverse %2 {axis = 1 : i32} : (tensor<34x83xf32>) -> tensor<34x83xf32>
    %4 = tosa.sub %3, %2 : (tensor<34x83xf32>, tensor<34x83xf32>) -> tensor<34x83xf32>
    %5 = tosa.exp %4 : (tensor<34x83xf32>) -> tensor<34x83xf32>
    %r_6 = tosa.const_shape {values = dense<[ 2822 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %5, %r_6 : (tensor<34x83xf32>, !tosa.shape<1>) -> tensor<2822xf32>
    %7 = tosa.abs %6 : (tensor<2822xf32>) -> tensor<2822xf32>
    return %7 : tensor<2822xf32>
  }
}
