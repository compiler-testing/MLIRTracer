module {
  func.func @main(%arg0: tensor<12x45x68x51x7xi32>, %arg1: tensor<12x1x68x51x7xi32>) -> tensor<12x45x68x51x7xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<12x45x68x51x7xi32>, tensor<12x1x68x51x7xi32>) -> tensor<12x45x68x51x7xi1>
    return %0 : tensor<12x45x68x51x7xi1>
  }
}
