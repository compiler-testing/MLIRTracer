module {
  func.func @main(%arg0: tensor<57x2x45x50x11x94xi32>) -> tensor<57x2x45x50x11x94xi32> {
    %0 = tosa.clamp %arg0 {min_val = -45 : i32, max_val = 21 : i32} : (tensor<57x2x45x50x11x94xi32>) -> tensor<57x2x45x50x11x94xi32>
    %1 = tosa.clz %0 : (tensor<57x2x45x50x11x94xi32>) -> tensor<57x2x45x50x11x94xi32>
    %2 = tosa.arithmetic_right_shift %1, %0 {round = true} : (tensor<57x2x45x50x11x94xi32>, tensor<57x2x45x50x11x94xi32>) -> tensor<57x2x45x50x11x94xi32>
    %3 = tosa.identity %2 : (tensor<57x2x45x50x11x94xi32>) -> tensor<57x2x45x50x11x94xi32>
    %4 = tosa.identity %3 : (tensor<57x2x45x50x11x94xi32>) -> tensor<57x2x45x50x11x94xi32>
    return %4 : tensor<57x2x45x50x11x94xi32>
  }
}
