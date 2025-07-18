module {
  func.func @main(%arg0: tensor<66x92x62x58x61x65xi32>, %arg1: tensor<66x92x1x1x1x1xi32>) -> tensor<66x92x62x58x61x65xi32> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<66x92x62x58x61x65xi32>, tensor<66x92x1x1x1x1xi32>) -> tensor<66x92x62x58x61x65xi32>
    %1 = tosa.clz %0 : (tensor<66x92x62x58x61x65xi32>) -> tensor<66x92x62x58x61x65xi32>
    return %1 : tensor<66x92x62x58x61x65xi32>
  }
}
