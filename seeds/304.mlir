module {
  func.func @main(%arg0: tensor<75x51x55x75x34x14xf32>, %arg1: tensor<58x66x61x100xi1>) -> (tensor<75x51x55x75x34x14xf32>, tensor<1x66x61x100xi1>) {
    %0 = tosa.log %arg0 : (tensor<75x51x55x75x34x14xf32>) -> tensor<75x51x55x75x34x14xf32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<58x66x61x100xi1>) -> tensor<1x66x61x100xi1>
    return %0, %1 : tensor<75x51x55x75x34x14xf32>, tensor<1x66x61x100xi1>
  }
}
