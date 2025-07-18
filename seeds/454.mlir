module {
  func.func @main(%arg0: tensor<53x56xi32>) -> tensor<56xi32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<53x56xi32>) -> tensor<56xi32>
    %r_1 = tosa.const_shape {values = dense<[ 56 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<56xi32>, !tosa.shape<1>) -> tensor<56xi32>
    %2 = tosa.clamp %1 {min_val = 55 : i32, max_val = 130 : i32} : (tensor<56xi32>) -> tensor<56xi32>
    return %2 : tensor<56xi32>
  }
}
