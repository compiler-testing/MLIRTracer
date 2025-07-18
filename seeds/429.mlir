module {
  func.func @main(%arg0: tensor<74xi64>, %arg1: tensor<74xi64>) -> tensor<74xi1> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<74xi64>, tensor<74xi64>) -> tensor<74xi64>
    %1 = tosa.greater %0, %0 : (tensor<74xi64>, tensor<74xi64>) -> tensor<74xi1>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<74xi1>, tensor<74xi1>) -> tensor<74xi1>
    %3 = tosa.clz %2 : (tensor<74xi1>) -> tensor<74xi1>
    return %3 : tensor<74xi1>
  }
}
