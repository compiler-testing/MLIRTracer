module {
  func.func @main(%arg0: tensor<50x79x72x50xi8>, %arg1: tensor<1x79x1x50xi8>) -> tensor<50x79x72x50xi1> {
    %0 = tosa.greater %arg0, %arg1 : (tensor<50x79x72x50xi8>, tensor<1x79x1x50xi8>) -> tensor<50x79x72x50xi1>
    %1 = tosa.identity %0 : (tensor<50x79x72x50xi1>) -> tensor<50x79x72x50xi1>
    %2 = tosa.identity %1 : (tensor<50x79x72x50xi1>) -> tensor<50x79x72x50xi1>
    return %2 : tensor<50x79x72x50xi1>
  }
}
