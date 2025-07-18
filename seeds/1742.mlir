module {
  func.func @main(%arg0: tensor<40xi32>, %arg1: tensor<40xi32>) -> tensor<40xi32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<40xi32>, tensor<40xi32>) -> tensor<40xi32>
    return %0 : tensor<40xi32>
  }
}
