module {
  func.func @main(%arg0: tensor<82x36x76x25xi32>) -> tensor<82x36x76x25xi32> {
    %0 = tosa.clamp %arg0 {min_val = -55 : i32, max_val = -33 : i32} : (tensor<82x36x76x25xi32>) -> tensor<82x36x76x25xi32>
    %1 = tosa.reverse %0 {axis = 2 : i32} : (tensor<82x36x76x25xi32>) -> tensor<82x36x76x25xi32>
    return %1 : tensor<82x36x76x25xi32>
  }
}
