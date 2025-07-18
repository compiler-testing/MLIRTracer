module {
  func.func @main(%arg0: tensor<48x11xi8>) -> tensor<48x11xi1> {
    %0 = tosa.identity %arg0 : (tensor<48x11xi8>) -> tensor<48x11xi8>
    %1 = tosa.greater_equal %0, %0 : (tensor<48x11xi8>, tensor<48x11xi8>) -> tensor<48x11xi1>
    return %1 : tensor<48x11xi1>
  }
}
