module {
  func.func @main(%arg0: tensor<71x63x13x87x55xi8>, %arg1: tensor<71x1x13x87x55xi8>, %arg2: tensor<24x77x99x82x42x71xi64>, %arg3: tensor<24x1x99x1x1x71xi64>, %arg4: tensor<52x41x13x79x67x98xf32>) -> (tensor<24x77x99x82x42x71xi1>, tensor<52x41x13x79x67x98xf32>, tensor<71x63x13x87x55xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<71x63x13x87x55xi8>, tensor<71x1x13x87x55xi8>) -> tensor<71x63x13x87x55xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<24x77x99x82x42x71xi64>, tensor<24x1x99x1x1x71xi64>) -> tensor<24x77x99x82x42x71xi1>
    %2 = tosa.floor %arg4 : (tensor<52x41x13x79x67x98xf32>) -> tensor<52x41x13x79x67x98xf32>
    %3 = tosa.logical_or %0, %0 : (tensor<71x63x13x87x55xi1>, tensor<71x63x13x87x55xi1>) -> tensor<71x63x13x87x55xi1>
    %4 = tosa.abs %3 : (tensor<71x63x13x87x55xi1>) -> tensor<71x63x13x87x55xi1>
    %5 = tosa.logical_left_shift %4, %3 : (tensor<71x63x13x87x55xi1>, tensor<71x63x13x87x55xi1>) -> tensor<71x63x13x87x55xi1>
    return %1, %2, %5 : tensor<24x77x99x82x42x71xi1>, tensor<52x41x13x79x67x98xf32>, tensor<71x63x13x87x55xi1>
  }
}
