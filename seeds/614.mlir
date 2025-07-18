module {
  func.func @main(%arg0: tensor<13x83x80x68xf32>, %arg1: tensor<4x2xi64>) -> (tensor<13x83x80x68xi1>, tensor<13x83x80x68xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<13x83x80x68xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<13x83x80x68xf32>
    %1 = tosa.maximum %0, %0 : (tensor<13x83x80x68xf32>, tensor<13x83x80x68xf32>) -> tensor<13x83x80x68xf32>
    %2 = tosa.equal %1, %0 : (tensor<13x83x80x68xf32>, tensor<13x83x80x68xf32>) -> tensor<13x83x80x68xi1>
    %3 = tosa.exp %0 : (tensor<13x83x80x68xf32>) -> tensor<13x83x80x68xf32>
    return %2, %3 : tensor<13x83x80x68xi1>, tensor<13x83x80x68xf32>
  }
}
