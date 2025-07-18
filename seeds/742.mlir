module {
  func.func @main(%arg0: tensor<48x81x80x52x63x44xf32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<48x81x80x52x63x44xf32>) {
    %0 = tosa.floor %arg0 : (tensor<48x81x80x52x63x44xf32>) -> tensor<48x81x80x52x63x44xf32>
    %1 = tosa.bitwise_not %arg1 : (tensor<i32>) -> tensor<i32>
    %2 = tosa.ceil %0 : (tensor<48x81x80x52x63x44xf32>) -> tensor<48x81x80x52x63x44xf32>
    return %1, %2 : tensor<i32>, tensor<48x81x80x52x63x44xf32>
  }
}
