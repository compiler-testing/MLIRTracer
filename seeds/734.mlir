module {
  func.func @main(%arg0: tensor<57x92x20xf32>, %arg1: tensor<12xi32>, %arg2: tensor<12xi32>) -> (tensor<2622x4x10xf32>, tensor<12xi32>) {
    %r_0 = tosa.const_shape {values = dense<[ 2622, 4, 10 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<57x92x20xf32>, !tosa.shape<3>) -> tensor<2622x4x10xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<12xi32>, tensor<12xi32>) -> tensor<12xi32>
    %2 = tosa.sigmoid %0 : (tensor<2622x4x10xf32>) -> tensor<2622x4x10xf32>
    %3 = tosa.reverse %1 {axis = 0 : i32} : (tensor<12xi32>) -> tensor<12xi32>
    return %2, %3 : tensor<2622x4x10xf32>, tensor<12xi32>
  }
}
