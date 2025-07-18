module {
  func.func @main(%arg0: tensor<78x9xi64>, %arg1: tensor<1x9xi64>, %arg2: tensor<1xf32>) -> (tensor<78x9xi64>, tensor<1xf32>, tensor<1x1xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<78x9xi64>, tensor<1x9xi64>) -> tensor<78x9xi64>
    %1 = tosa.reciprocal %arg2 : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.minimum %0, %0 : (tensor<78x9xi64>, tensor<78x9xi64>) -> tensor<78x9xi64>
    %3 = tosa.log %1 : (tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.abs %1 : (tensor<1xf32>) -> tensor<1xf32>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %3, %r_5 : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    %6 = tosa.clamp %5 {min_val = -1.600000e+01 : f32, max_val = 9.300000e+01 : f32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    return %2, %4, %6 : tensor<78x9xi64>, tensor<1xf32>, tensor<1x1xf32>
  }
}
