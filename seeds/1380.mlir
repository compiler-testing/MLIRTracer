module {
  func.func @main(%arg0: tensor<70x58x23x52x55x39xi8>, %arg1: tensor<1x58x1x52x55x39xi8>, %arg2: tensor<52x88x91x40xi1>) -> (tensor<70x58x23x52x55x39xi8>, tensor<52x88x91x40xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<70x58x23x52x55x39xi8>, tensor<1x58x1x52x55x39xi8>) -> tensor<70x58x23x52x55x39xi8>
    %1 = tosa.logical_not %arg2 : (tensor<52x88x91x40xi1>) -> tensor<52x88x91x40xi1>
    %2 = tosa.sub %1, %1 : (tensor<52x88x91x40xi1>, tensor<52x88x91x40xi1>) -> tensor<52x88x91x40xi1>
    %3 = tosa.sub %2, %2 : (tensor<52x88x91x40xi1>, tensor<52x88x91x40xi1>) -> tensor<52x88x91x40xi1>
    %4 = tosa.bitwise_and %3, %1 : (tensor<52x88x91x40xi1>, tensor<52x88x91x40xi1>) -> tensor<52x88x91x40xi1>
    return %0, %4 : tensor<70x58x23x52x55x39xi8>, tensor<52x88x91x40xi1>
  }
}
