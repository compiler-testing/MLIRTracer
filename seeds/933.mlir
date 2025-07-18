module {
  func.func @main(%arg0: tensor<35x2x94x43x59x2xi1>, %arg1: tensor<1x2x94x43x1x1xi1>, %arg2: tensor<60x40x10x72x43xi64>, %arg3: tensor<1x40x1x1x1xi64>) -> (tensor<35x2x94x43x59x2xi1>, tensor<60x40x10x72x43xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<35x2x94x43x59x2xi1>, tensor<1x2x94x43x1x1xi1>) -> tensor<35x2x94x43x59x2xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<60x40x10x72x43xi64>, tensor<1x40x1x1x1xi64>) -> tensor<60x40x10x72x43xi1>
    return %0, %1 : tensor<35x2x94x43x59x2xi1>, tensor<60x40x10x72x43xi1>
  }
}
