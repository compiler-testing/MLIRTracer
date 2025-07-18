module {
  func.func @main(%arg0: tensor<85x52x53x95x49xi1>, %arg1: tensor<85x1x1x1x49xi1>, %arg2: tensor<50x58x73xf32>, %arg3: tensor<f32>) -> (tensor<50x58xi32>, tensor<85x52x53x95x49xi1>, tensor<f32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<85x52x53x95x49xi1>, tensor<85x1x1x1x49xi1>) -> tensor<85x52x53x95x49xi1>
    %1 = tosa.argmax %arg2 {axis = 2 : i32} : (tensor<50x58x73xf32>) -> tensor<50x58xi32>
    %2 = tosa.clz %0 : (tensor<85x52x53x95x49xi1>) -> tensor<85x52x53x95x49xi1>
    %3 = tosa.clz %2 : (tensor<85x52x53x95x49xi1>) -> tensor<85x52x53x95x49xi1>
    %4 = tosa.sigmoid %arg3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.intdiv %1, %1 : (tensor<50x58xi32>, tensor<50x58xi32>) -> tensor<50x58xi32>
    %6 = tosa.logical_xor %3, %0 : (tensor<85x52x53x95x49xi1>, tensor<85x52x53x95x49xi1>) -> tensor<85x52x53x95x49xi1>
    %7 = tosa.log %4 : (tensor<f32>) -> tensor<f32>
    return %5, %6, %7 : tensor<50x58xi32>, tensor<85x52x53x95x49xi1>, tensor<f32>
  }
}
