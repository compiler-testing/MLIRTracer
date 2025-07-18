module {
  func.func @main(%arg0: tensor<69x91x46x72x40xi32>, %arg1: tensor<69x1x1x72x1xi32>, %arg2: tensor<59x49x88xf32>, %arg3: tensor<81x98x31x58xf32>) -> (tensor<69x91x46x72x40xi1>, tensor<49x88xi1>, tensor<81x98x31x58xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<69x91x46x72x40xi32>, tensor<69x1x1x72x1xi32>) -> tensor<69x91x46x72x40xi32>
    %1 = tosa.greater_equal %0, %0 : (tensor<69x91x46x72x40xi32>, tensor<69x91x46x72x40xi32>) -> tensor<69x91x46x72x40xi1>
    %2 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<59x49x88xf32>) -> tensor<49x88xi32>
    %3 = tosa.equal %2, %2 : (tensor<49x88xi32>, tensor<49x88xi32>) -> tensor<49x88xi1>
    %4 = tosa.add %3, %3 : (tensor<49x88xi1>, tensor<49x88xi1>) -> tensor<49x88xi1>
    %5 = tosa.log %arg3 : (tensor<81x98x31x58xf32>) -> tensor<81x98x31x58xf32>
    return %1, %4, %5 : tensor<69x91x46x72x40xi1>, tensor<49x88xi1>, tensor<81x98x31x58xf32>
  }
}
