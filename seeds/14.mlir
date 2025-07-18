module {
  func.func @main(%arg0: tensor<62x79x21x52x49xf32>) -> tensor<62x79x21x52x49xf32> {
    %0 = tosa.floor %arg0 : (tensor<62x79x21x52x49xf32>) -> tensor<62x79x21x52x49xf32>
    %1 = tosa.pow %0, %0 : (tensor<62x79x21x52x49xf32>, tensor<62x79x21x52x49xf32>) -> tensor<62x79x21x52x49xf32>
    return %1 : tensor<62x79x21x52x49xf32>
  }
}
