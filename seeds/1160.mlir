module {
  func.func @main(%arg0: tensor<89x53x92x38x24xf32>, %arg1: tensor<1x1x1x1x24xf32>, %arg2: tensor<87x73xi8>, %arg3: tensor<22x33x30xi1>) -> (tensor<89x53x92x38x24xf32>, tensor<22x33x1xi1>, tensor<1xi32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<89x53x92x38x24xf32>, tensor<1x1x1x1x24xf32>) -> tensor<89x53x92x38x24xf32>
    %1 = tosa.argmax %arg2 {axis = 1 : i32} : (tensor<87x73xi8>) -> tensor<87xi32>
    %2 = tosa.reduce_any %arg3 {axis = 2 : i32} : (tensor<22x33x30xi1>) -> tensor<22x33x1xi1>
    %3 = tosa.reverse %1 {axis = 0 : i32} : (tensor<87xi32>) -> tensor<87xi32>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<87xi32>) -> tensor<1xi32>
    return %0, %2, %4 : tensor<89x53x92x38x24xf32>, tensor<22x33x1xi1>, tensor<1xi32>
  }
}
