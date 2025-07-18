module {
  func.func @main(%arg0: tensor<49x49xf32>, %arg1: tensor<69xi16>) -> (tensor<69xi16>, tensor<49x49xf32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<49x49xf32>) -> tensor<49x49xf32>
    %1 = tosa.clz %arg1 : (tensor<69xi16>) -> tensor<69xi16>
    %2 = tosa.ceil %0 : (tensor<49x49xf32>) -> tensor<49x49xf32>
    %3 = tosa.sigmoid %2 : (tensor<49x49xf32>) -> tensor<49x49xf32>
    return %1, %3 : tensor<69xi16>, tensor<49x49xf32>
  }
}
