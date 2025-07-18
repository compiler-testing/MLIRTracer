module {
  func.func @main(%arg0: tensor<44x51x92x82x43x9xi64>, %arg1: tensor<44x1x92x1x1x9xi64>) -> tensor<44x51x92x82x43x9xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<44x51x92x82x43x9xi64>, tensor<44x1x92x1x1x9xi64>) -> tensor<44x51x92x82x43x9xi1>
    %1 = tosa.bitwise_not %0 : (tensor<44x51x92x82x43x9xi1>) -> tensor<44x51x92x82x43x9xi1>
    return %1 : tensor<44x51x92x82x43x9xi1>
  }
}
