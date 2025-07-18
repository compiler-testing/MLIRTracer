module {
  func.func @main(%arg0: tensor<100x79xi64>, %arg1: tensor<100x1xi64>, %arg2: tensor<73x37x9xi32>, %arg3: tensor<1x1x1xi32>, %arg4: tensor<88x41x39x28x11x13xf32>) -> (tensor<100x79xi1>, tensor<88x41x39x28x11x13xf32>, tensor<73x37xi32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<100x79xi64>, tensor<100x1xi64>) -> tensor<100x79xi64>
    %1 = tosa.equal %0, %0 : (tensor<100x79xi64>, tensor<100x79xi64>) -> tensor<100x79xi1>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<73x37x9xi32>, tensor<1x1x1xi32>) -> tensor<73x37x9xi32>
    %3 = tosa.abs %2 : (tensor<73x37x9xi32>) -> tensor<73x37x9xi32>
    %4 = tosa.logical_or %1, %1 : (tensor<100x79xi1>, tensor<100x79xi1>) -> tensor<100x79xi1>
    %5 = tosa.logical_left_shift %4, %1 : (tensor<100x79xi1>, tensor<100x79xi1>) -> tensor<100x79xi1>
    %6 = tosa.log %arg4 : (tensor<88x41x39x28x11x13xf32>) -> tensor<88x41x39x28x11x13xf32>
    %7 = tosa.pow %6, %6 : (tensor<88x41x39x28x11x13xf32>, tensor<88x41x39x28x11x13xf32>) -> tensor<88x41x39x28x11x13xf32>
    %8 = tosa.argmax %3 {axis = 2 : i32} : (tensor<73x37x9xi32>) -> tensor<73x37xi32>
    return %5, %7, %8 : tensor<100x79xi1>, tensor<88x41x39x28x11x13xf32>, tensor<73x37xi32>
  }
}
