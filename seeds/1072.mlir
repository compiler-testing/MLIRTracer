module {
  func.func @main(%arg0: tensor<25x40x5x36x27x34xf32>, %arg1: tensor<25x1x1x1x1x34xf32>) -> tensor<459000x360xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<25x40x5x36x27x34xf32>, tensor<25x1x1x1x1x34xf32>) -> tensor<25x40x5x36x27x34xf32>
    %r_1 = tosa.const_shape {values = dense<[ 459000, 360 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<25x40x5x36x27x34xf32>, !tosa.shape<2>) -> tensor<459000x360xf32>
    %2 = tosa.clamp %1 {min_val = -5.000000e+00 : f32, max_val = 3.100000e+01 : f32} : (tensor<459000x360xf32>) -> tensor<459000x360xf32>
    return %2 : tensor<459000x360xf32>
  }
}
