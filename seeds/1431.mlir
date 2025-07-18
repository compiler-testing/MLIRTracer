module {
  func.func @main(%arg0: tensor<51x94x86x63xf32>, %arg1: tensor<11x35xi32>, %arg2: tensor<1x1xi32>) -> (tensor<51x94x1x63xf32>, tensor<11x35xi32>, tensor<11x35xi32>) {
    %0 = tosa.tanh %arg0 : (tensor<51x94x86x63xf32>) -> tensor<51x94x86x63xf32>
    %1 = tosa.ceil %0 : (tensor<51x94x86x63xf32>) -> tensor<51x94x86x63xf32>
    %2 = tosa.ceil %1 : (tensor<51x94x86x63xf32>) -> tensor<51x94x86x63xf32>
    %3 = tosa.abs %2 : (tensor<51x94x86x63xf32>) -> tensor<51x94x86x63xf32>
    %4 = tosa.reduce_max %3 {axis = 2 : i32} : (tensor<51x94x86x63xf32>) -> tensor<51x94x1x63xf32>
    %5 = tosa.intdiv %arg1, %arg2 : (tensor<11x35xi32>, tensor<1x1xi32>) -> tensor<11x35xi32>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<11x35xi32>, tensor<11x35xi32>) -> tensor<11x35xi32>
    %7 = tosa.bitwise_not %5 : (tensor<11x35xi32>) -> tensor<11x35xi32>
    return %4, %6, %7 : tensor<51x94x1x63xf32>, tensor<11x35xi32>, tensor<11x35xi32>
  }
}
