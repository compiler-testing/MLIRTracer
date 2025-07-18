module {
  func.func @main(%arg0: tensor<49x73x34x37x42x2xi16>, %arg1: tensor<49x1x1x37x42x2xi16>, %arg2: tensor<42x5xf32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<49x73x34x37x42x2xi16>, tensor<42x1xf32>, tensor<42x5xf32>, tensor<i32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<49x73x34x37x42x2xi16>, tensor<49x1x1x37x42x2xi16>) -> tensor<49x73x34x37x42x2xi16>
    %1 = tosa.ceil %arg2 : (tensor<42x5xf32>) -> tensor<42x5xf32>
    %2 = tosa.sigmoid %1 : (tensor<42x5xf32>) -> tensor<42x5xf32>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<42x5xf32>) -> tensor<42x1xf32>
    %4 = tosa.sub %1, %2 : (tensor<42x5xf32>, tensor<42x5xf32>) -> tensor<42x5xf32>
    %5 = tosa.intdiv %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0, %3, %4, %6 : tensor<49x73x34x37x42x2xi16>, tensor<42x1xf32>, tensor<42x5xf32>, tensor<i32>
  }
}
